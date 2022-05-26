import os

from cairosvg import svg2png

if __name__ == '__main__':
    path = 'samples/'
    prefix = 'samples'
    for image in list(os.listdir(path)):
        foldername = prefix + '-' + image.split('-')[0]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        with open(os.path.join(path, image), 'r') as f:
            svg2png(file_obj=f, write_to=os.path.join(foldername, image).replace('svg', 'png'), dpi=300,
                    output_width=500, output_height=500)
