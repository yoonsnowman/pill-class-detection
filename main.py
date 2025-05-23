from hd import run_hd
from mk import run_mk
from dw import run_dw
from ch import run_ch


def main():
    x = 5
    print(f"설정값: {x}")
    x = run_hd(x)
    print(f"현도 통과: {x}")
    x = run_mk(x)
    print(f"민경 통과: {x}")
    x = run_dw(x)
    print(f"동우 통과: {x}")
    x = run_ch(x)
    print(f"창훈 통과: {x}")

    print(f"최종값: {x}")

if __name__ == "__main__":
    main()