"""
PCAP to Features Extractor
===========================
Reads a PCAP file captured from a live network, groups packets into
bidirectional TCP/UDP flows (5-tuple) with idle-timeout-based splitting,
computes the same 10 features used by our XGBoost DoS detection model,
scales them with the saved StandardScaler, and writes a CSV ready for
the dashboard.

Features computed (in model order):
    rate, sload, sbytes, dload, proto, dtcpb, stcpb, dmean, tcprtt, dur

Usage:
    python pcap_to_features.py capture.pcap

Outputs:
    live_traffic_scaled.csv   — 10 columns, scaled, for dashboard upload
    live_traffic_raw.csv      — raw values + metadata for inspection
"""

import sys
import pickle
import csv
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "03_model_training" / "proper_training"
SCALER_PATH = MODEL_DIR / "data" / "feature_scaler.pkl"
ENCODER_PATH = MODEL_DIR / "data" / "proto_encoder.pkl"

FEATURE_NAMES = ["rate", "sload", "sbytes", "dload", "proto",
                 "dtcpb", "stcpb", "dmean", "tcprtt", "dur"]

# If a flow has no packets for this many seconds, close it and start fresh.
# This prevents port-reuse from merging unrelated connections.
FLOW_IDLE_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Flow tracker
# ---------------------------------------------------------------------------
class Flow:
    """Tracks packets belonging to a single bidirectional connection."""

    __slots__ = [
        "src_ip", "src_port", "dst_ip", "dst_port", "proto_num",
        "first_ts", "last_ts",
        "fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts",
        "syn_ts", "synack_ts",
        "fwd_first_seq", "bwd_first_seq",
    ]

    def __init__(self, src_ip, src_port, dst_ip, dst_port, proto_num, ts):
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.proto_num = proto_num
        self.first_ts = ts
        self.last_ts = ts
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_pkts = 0
        self.bwd_pkts = 0
        self.syn_ts = None
        self.synack_ts = None
        self.fwd_first_seq = None
        self.bwd_first_seq = None

    def add_forward(self, ts, ip_len, tcp_flags=0, tcp_seq=0):
        self.last_ts = max(self.last_ts, ts)
        self.fwd_bytes += ip_len
        self.fwd_pkts += 1
        if self.fwd_first_seq is None and tcp_seq:
            self.fwd_first_seq = tcp_seq
        # SYN only (not SYN-ACK)
        if (tcp_flags & 0x02) and not (tcp_flags & 0x10):
            if self.syn_ts is None:
                self.syn_ts = ts

    def add_backward(self, ts, ip_len, tcp_flags=0, tcp_seq=0):
        self.last_ts = max(self.last_ts, ts)
        self.bwd_bytes += ip_len
        self.bwd_pkts += 1
        if self.bwd_first_seq is None and tcp_seq:
            self.bwd_first_seq = tcp_seq
        # SYN-ACK
        if (tcp_flags & 0x12) == 0x12:
            if self.synack_ts is None:
                self.synack_ts = ts

    def to_features(self):
        """Compute the 10 model features from this flow."""
        dur = self.last_ts - self.first_ts

        total_pkts = self.fwd_pkts + self.bwd_pkts

        # Handle zero-duration flows (single packet): set rate/load to 0
        # rather than infinity — matches how Argus/Bro handle them.
        if dur <= 0:
            rate = 0.0
            sload = 0.0
            dload = 0.0
        else:
            rate = total_pkts / dur
            sload = self.fwd_bytes / dur
            dload = self.bwd_bytes / dur

        sbytes = self.fwd_bytes
        stcpb = self.fwd_first_seq if self.fwd_first_seq is not None else 0
        dtcpb = self.bwd_first_seq if self.bwd_first_seq is not None else 0
        dmean = (self.bwd_bytes / self.bwd_pkts) if self.bwd_pkts > 0 else 0.0

        tcprtt = 0.0
        if self.syn_ts is not None and self.synack_ts is not None:
            rtt = self.synack_ts - self.syn_ts
            if 0 < rtt < 10.0:  # sanity: RTT should be < 10 seconds
                tcprtt = rtt

        return {
            "rate": rate,
            "sload": sload,
            "sbytes": float(sbytes),
            "dload": dload,
            "proto": self.proto_num,  # will be encoded later
            "dtcpb": float(dtcpb),
            "stcpb": float(stcpb),
            "dmean": dmean,
            "tcprtt": tcprtt,
            "dur": dur,
        }

    def metadata(self):
        return {
            "src": f"{self.src_ip}:{self.src_port}",
            "dst": f"{self.dst_ip}:{self.dst_port}",
            "proto_num": self.proto_num,
            "fwd_pkts": self.fwd_pkts,
            "bwd_pkts": self.bwd_pkts,
            "start_time": round(self.first_ts, 4),
            "end_time": round(self.last_ts, 4),
        }


# ---------------------------------------------------------------------------
# Flow extraction with idle-timeout splitting
# ---------------------------------------------------------------------------
def extract_flows(pcap_path):
    """
    Read PCAP, group packets into bidirectional flows.
    If a flow is idle for > FLOW_IDLE_TIMEOUT seconds, close it and
    start a new one (prevents port reuse from merging unrelated flows).
    """
    from scapy.all import PcapReader, IP, TCP, UDP

    active = {}           # 5-tuple key -> Flow (currently active flows)
    completed = []        # list of finished Flow objects
    pkt_count = 0
    first_global_ts = None

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500_000 == 0:
                print(f"  ... processed {pkt_count:,} packets")

            if not pkt.haslayer(IP):
                continue

            ip = pkt[IP]
            ts = float(pkt.time)
            proto_num = ip.proto

            if first_global_ts is None:
                first_global_ts = ts

            src_ip = ip.src
            dst_ip = ip.dst
            ip_len = ip.len        # total IP-level bytes (header + payload)
            tcp_flags = 0
            tcp_seq = 0
            sport = 0
            dport = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                sport = tcp.sport
                dport = tcp.dport
                tcp_flags = int(tcp.flags)
                tcp_seq = tcp.seq
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                sport = udp.sport
                dport = udp.dport

            fwd_key = (src_ip, sport, dst_ip, dport, proto_num)
            bwd_key = (dst_ip, dport, src_ip, sport, proto_num)

            # Check if existing flow has timed out
            matched_key = None
            is_forward = True

            if fwd_key in active:
                if ts - active[fwd_key].last_ts > FLOW_IDLE_TIMEOUT:
                    completed.append(active.pop(fwd_key))
                else:
                    matched_key = fwd_key
                    is_forward = True

            if matched_key is None and bwd_key in active:
                if ts - active[bwd_key].last_ts > FLOW_IDLE_TIMEOUT:
                    completed.append(active.pop(bwd_key))
                else:
                    matched_key = bwd_key
                    is_forward = False

            if matched_key is not None:
                flow = active[matched_key]
                if is_forward:
                    flow.add_forward(ts, ip_len, tcp_flags, tcp_seq)
                else:
                    flow.add_backward(ts, ip_len, tcp_flags, tcp_seq)
            else:
                # New flow
                flow = Flow(src_ip, sport, dst_ip, dport, proto_num, ts)
                flow.add_forward(ts, ip_len, tcp_flags, tcp_seq)
                active[fwd_key] = flow

    # Close all remaining active flows
    completed.extend(active.values())

    print(f"  Total packets: {pkt_count:,}")
    print(f"  Total flows: {len(completed):,}")
    return completed, first_global_ts


# ---------------------------------------------------------------------------
# Proto encoding
# ---------------------------------------------------------------------------
PROTO_NUM_TO_NAME = {
    6: "tcp", 17: "udp", 1: "icmp", 2: "igmp", 47: "gre",
    41: "ipv6", 89: "ospf", 132: "sctp", 50: "esp", 51: "ah",
}


def encode_proto(proto_num, encoder):
    name = PROTO_NUM_TO_NAME.get(proto_num, str(proto_num))
    try:
        return encoder.transform([name])[0]
    except ValueError:
        return encoder.transform(["tcp"])[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python pcap_to_features.py <capture.pcap>")
        sys.exit(1)

    pcap_path = Path(sys.argv[1])
    if not pcap_path.exists():
        print(f"Error: {pcap_path} not found")
        sys.exit(1)

    print("Loading scaler and encoder...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    print(f"Reading {pcap_path} ...")
    flows, first_ts = extract_flows(pcap_path)

    if not flows:
        print("No flows found in PCAP.")
        sys.exit(1)

    print(f"Computing features for {len(flows):,} flows...")
    raw_rows = []
    for flow in flows:
        feat = flow.to_features()
        meta = flow.metadata()
        feat["proto"] = encode_proto(flow.proto_num, encoder)
        raw_rows.append({**feat, **meta})

    raw_rows.sort(key=lambda r: r["start_time"])

    # Write raw CSV
    raw_csv_path = pcap_path.parent / "live_traffic_raw.csv"
    raw_fields = FEATURE_NAMES + ["src", "dst", "proto_num", "fwd_pkts",
                                   "bwd_pkts", "start_time", "end_time"]
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({k: row[k] for k in raw_fields})
    print(f"  Raw features: {raw_csv_path}")

    # Scale features
    feature_matrix = np.array([[row[f] for f in FEATURE_NAMES] for row in raw_rows])
    scaled_matrix = scaler.transform(feature_matrix)

    # Write scaled CSV
    scaled_csv_path = pcap_path.parent / "live_traffic_scaled.csv"
    with open(scaled_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES)
        for row_vals in scaled_matrix:
            writer.writerow([f"{v:.8f}" for v in row_vals])
    print(f"  Scaled features: {scaled_csv_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total flows: {len(raw_rows):,}")

    dst_counter = Counter(r["dst"].split(":")[0] for r in raw_rows)
    print(f"\nFlows by destination IP:")
    for ip, count in dst_counter.most_common(10):
        print(f"  {ip}: {count:,} flows")

    # Categorize flows
    flood_start = 200.0   # seconds from capture start (from our analysis)
    n_normal = sum(1 for r in raw_rows
                   if r["start_time"] - first_ts < flood_start
                   or r["dst"].split(":")[0] != "10.0.2.19")
    n_attack = len(raw_rows) - n_normal
    print(f"\nEstimated Normal flows (before t=200s or to other IPs): {n_normal:,}")
    print(f"Estimated Attack flows (after t=200s to 10.0.2.19): {n_attack:,}")

    # Separate stats for normal vs attack
    normal_indices = [i for i, r in enumerate(raw_rows)
                      if r["start_time"] - first_ts < flood_start
                      or r["dst"].split(":")[0] != "10.0.2.19"]
    attack_indices = [i for i, r in enumerate(raw_rows)
                      if i not in set(normal_indices)]

    for label, indices in [("NORMAL flows", normal_indices),
                           ("ATTACK flows", attack_indices)]:
        if not indices:
            continue
        sub = feature_matrix[indices]
        sub_s = scaled_matrix[indices]
        print(f"\n--- {label} ({len(indices):,} flows) ---")
        print(f"  {'Feature':>8s}  {'raw_mean':>12s}  {'raw_max':>12s}  "
              f"{'scaled_mean':>12s}  {'scaled_max':>12s}")
        for i, name in enumerate(FEATURE_NAMES):
            print(f"  {name:>8s}  {sub[:,i].mean():>12.2f}  {sub[:,i].max():>12.2f}  "
                  f"{sub_s[:,i].mean():>12.3f}  {sub_s[:,i].max():>12.3f}")

    # UNSW-NB15 scaler reference
    print(f"\n--- UNSW-NB15 scaler reference (mean / std) ---")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:>8s}:  mean={scaler.mean_[i]:.2f}  std={np.sqrt(scaler.var_[i]):.2f}")

    print(f"\n{'='*60}")
    print(f"Upload '{scaled_csv_path.name}' to the dashboard Analyze page")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
