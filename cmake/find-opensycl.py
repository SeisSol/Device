import os
import sys
from shutil import which
import argparse
import json


def get_package_dir():
  exe_name = which("syclcc")
  if not exe_name:
    sys.exit(1)

  bin_dir = os.path.dirname(exe_name)
  return os.path.dirname(bin_dir)


def get_config():
  package_dir = get_package_dir()
  etc_dir = os.path.join(package_dir, "etc")
  config_file = None
  for root, dirs, files in os.walk(etc_dir):
    for file in files:
      if file == "syclcc.json":
        config_file = os.path.join(root, file)
        break
  
  if not config_file:
    print("failed to find Open SYCL config file", file=sys.stderr)
    sys.exit(1)
  
  with open(config_file, 'r') as file:
    data = json.load(file)
  return data


def get_full_target_name(arch):
  data = get_config()
  platform = data["default-platform"]
  if platform == 'cuda':
    if not (data['default-nvcxx'].isspace() or 'NOTFOUND' in data['default-nvcxx']):
      default_target='cuda-nvcxx'
      arch = arch.replace('sm_', 'cc')
    else:
      default_target='cuda'
  elif platform == 'rocm':
    default_target='hip'
  print(f'{default_target}:{arch}', end='')


def get_vendor_name():
  data = get_config()
  platform = data["default-platform"]
  if platform == 'cuda':
    print('NVIDIA', end='')
  elif platform == 'rocm':
    print('AMD', end='')
  else:
    print('Intel', end='')


def get_include_dir():
  package_dir = get_package_dir()
  include_dir = os.path.join(package_dir, 'include')
  print(include_dir, end = '')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Extracts info from opensycl')
  parser.add_argument('-t', '--target', action='store_true', help='get default target')
  parser.add_argument('-i', '--include_dir', action='store_true', help='get include dir')
  parser.add_argument('-a', '--arch', type=str, help='architecture name')
  parser.add_argument('--vendor', action='store_true', help='get vendor name')
  args = parser.parse_args()

  if (args.target):
    if (not args.arch):
      print("architecture name is not provided", file=sys.stderr)
      sys.exit(1)
    get_full_target_name(args.arch)
  
  if (args.include_dir):
    get_include_dir()

  if (args.vendor):
    get_vendor_name()
