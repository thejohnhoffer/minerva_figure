''' for debugging script
'''
from minerva_scripts.scripts import combine

ROOT = '/home/j/2018/minerva-scripts/'
CONFIG_FILE = ROOT + 'examples/combine_ashlar_4channels.yaml'
INPUT_FOLDER = '/media/j/420D-AC8E/cycif_images/40BP_59/tiles/'
OUTPUT_FOLDER = ROOT + 'examples/tmp/'

combine.main([CONFIG_FILE, '-i', INPUT_FOLDER, '-o', OUTPUT_FOLDER])
