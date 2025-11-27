from PIL import Image


def create_gifs(images, name):

    # create the frames
    frames = []
    imgs = images
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    rev = frames[::-1]
    rev.pop(0)
    rev.pop(-1)

    frames += rev

    frames[0].save("rickroll/{}.gif".format(name), format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=1000//10, loop=0)


cnts = list(range(1, 25))

image_names = ['results/datadance_250_{}_datadance_250_{}/datadance_250_{}-image-00099.png'.format(cnt, cnt, cnt) for cnt in cnts]

create_gifs(image_names, 'datadance250_24-11')
