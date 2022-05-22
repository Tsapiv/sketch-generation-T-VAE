import os, shutil

from cairosvg import svg2png

# svg_code = """
#     <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
#         <circle cx="12" cy="12" r="10"/>
#         <line x1="12" y1="8" x2="12" y2="12"/>
#         <line x1="12" y1="16" x2="12" y2="16"/>
#     </svg>
# """
#
# svg2png(bytestring=svg_code,write_to='output.png')

if __name__ == '__main__':
    path = 'samples_trans_conditional/'
    prefix = 'trans_conditional_'
    for image in list(os.listdir(path)):
        foldername = prefix + image.split('_')[0]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        with open(os.path.join(path, image), 'r') as f:
            svg2png(file_obj=f, write_to=os.path.join(foldername, image).replace('svg', 'png'), dpi=200, output_width=100, output_height=100)
        # shutil.copy(os.path.join(path, image), os.path.join(foldername, image))