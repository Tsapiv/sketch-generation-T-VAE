import os

from cairosvg import svg2png

if __name__ == '__main__':
    path = 'samples_trans_conditional/'
    prefix = 'trans_conditional_'
    for image in list(os.listdir(path)):
        foldername = prefix + image.split('_')[0]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        with open(os.path.join(path, image), 'r') as f:
            svg2png(file_obj=f, write_to=os.path.join(foldername, image).replace('svg', 'png'), dpi=300,
                    output_width=100, output_height=100)
