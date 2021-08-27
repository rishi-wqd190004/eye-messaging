import argparse
import os
import sys
import shutil
import json

'''
ARGS:
argParse with input file path
argParse with output file path

Return:

'''
parser = argparse.ArgumentParser(description="eye tracker prepping dataset")
parser.add_argument('--dataset_path', help="Path to the extracted folder for CVPR 2016")
parser.add_argument('--output_path', default=None, help="Output folder for writing the results")
args = parser.parse_args()

def main():
    '''
    reading the json files,
    preparing the path for the input files
    cropping the images to get eyes only


    '''
    if args.output_path is None:
        args.output_path = args.dataset_path

    if args.dataset_path is None:
        raise RuntimeError("No dataset_path specified % " % args.dataset_path)
    
    preparePath(args.output_path)

    
      

def readJson(filename):
    if not os.path.isfile(filename):
        loggingError('Warning: No such file %s !' % filename)
        return None
    
    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None
        
    if data is None:
        loggingError("Warning! Couldn't read the files %s" % filename)
        return None
    return data
    

def preparePath(path, clear=False):
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)
    return path

def loggingError(msg, critical=False):
    print(msg)
    if critical:
        sys.exit(1)

if __name__ == '__main__':
    main()
    print('Done preparing of your dataset')