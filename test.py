import sys
import time

from hardware.ledalert import send_alert


def main() -> None:
    if len(sys.argv) > 1:
        level = sys.argv[1].upper()
        send_alert(level)
        print(f"Sent {level} to Arduino.")
        return

    send_alert("HIGH")
    time.sleep(2)

    send_alert("MEDIUM")
    time.sleep(2)

    send_alert("LOW")
    print("Sent HIGH, MEDIUM, LOW to Arduino.")


if __name__ == "__main__":
    main()
