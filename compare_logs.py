"""并排对比缩放前后的 simai token 文件"""
from pathlib import Path

SEP = "-" * 50
log_dir = Path("logs")

for sid, diff_name in [("100", "Advanced"), ("10", "Master")]:
    orig = (log_dir / f"{sid}_{diff_name}_orig.txt").read_text(encoding="utf-8").split("\n")
    scaled = (log_dir / f"{sid}_{diff_name}_scaled.txt").read_text(encoding="utf-8").split("\n")

    print(f"\n{'='*80}")
    print(f"  [{sid}] {diff_name}  --  original vs scaled(to subdiv=4)")
    print(f"{'='*80}")

    # 取前25行对比
    for i in range(min(25, max(len(orig), len(scaled)))):
        o = orig[i] if i < len(orig) else ""
        s = scaled[i] if i < len(scaled) else ""

        flag = ""
        if i > 0:
            o_parts = o.split()
            s_parts = s.split()
            if len(o_parts) >= 3 and len(s_parts) >= 3:
                if o_parts[2] != s_parts[2]:
                    flag = "  <<< subdiv CHANGED"

        print(f"  {o:<50s} | {s:<50s}{flag}")
