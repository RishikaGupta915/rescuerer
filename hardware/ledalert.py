import time
try:
    import serial
except ImportError as exc:
    serial = None
    _serial_import_error = exc
else:
    _serial_import_error = None

SERIAL_PORT = "COM9"
BAUD_RATE = 9600

arduino = None


def _get_arduino():
    global arduino

    if serial is None:
        raise RuntimeError(
            "pyserial is not installed. Run '.\\venv\\Scripts\\python.exe -m pip install pyserial'."
        ) from _serial_import_error

    if arduino is None:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)

    return arduino


def close_arduino():
    global arduino

    if arduino is None:
        return

    try:
        arduino.close()
    finally:
        arduino = None

def send_alert(level):
    board = _get_arduino()

    if level == "HIGH":
        board.write(b'H')

    elif level == "MEDIUM":
        board.write(b'M')

    elif level == "LOW":
        board.write(b'L')

    elif level in {"OFF", "NONE"}:
        board.write(b'N')

    else:
        raise ValueError(f"Unsupported alert level: {level}")
