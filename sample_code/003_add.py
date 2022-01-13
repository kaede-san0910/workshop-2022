import argparse

parser = argparse.ArgumentParser(description='2つの数字を足す')
parser.add_argument('a', type=int, help='１つ目の数字')
parser.add_argument('b', type=int, help='２つ目の数字')
args = parser.parse_args()

print("a = {}".format(args.a))
print("b = {}".format(args.b))
print("{} + {} = {}".format(args.a, args.b, args.a+args.b))