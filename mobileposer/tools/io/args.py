import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="imuposer combo training")
    parser.add_argument("--combo_id", help="combo id", type=str, required=True)
    
    return parser