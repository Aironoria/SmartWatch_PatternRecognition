import os

from PIL import Image, ImageDraw, ImageFont



def draw_arrow():
    arrow_size = 50
    rect_width = 50
    rect_height = 150

    image = Image.new('RGB', (640, 640), (0, 0, 0))

    # 在图像中心添加一个红色的向下箭头和箭头屁股
    draw = ImageDraw.Draw(image)
    arrow_x, arrow_y = (320, 395)
    draw.polygon([(arrow_x - arrow_size, arrow_y - arrow_size),
                  (arrow_x + arrow_size, arrow_y - arrow_size),
                  (arrow_x, arrow_y + arrow_size)],
                 fill=(255, 0, 0))
    #200 , 450
    # 绘制矩形
    draw.rectangle([(arrow_x - rect_width / 2, arrow_y - arrow_size - rect_height),
                    (arrow_x + rect_width / 2, arrow_y - arrow_size)],
                   fill=(255, 0, 0))
    return image

def crop(image):
    left = 120
    upper = 0
    right = 120+400
    lower = 640
    box = (left, upper, right, lower)

    # 剪裁图片
    return image.crop(box)

# 绘制向上箭头
def draw_arrow_up(filename):
    image = draw_arrow()
    image = image.rotate(180)
    image = crop(image)
    image.save(filename+ '.png')

def draw_arrow_down(filename):
    image = draw_arrow()
    image = crop(image)
    image.save(filename+'.png')
def draw_arrow_left(filename):
    image = draw_arrow()
    image = image.rotate(-90)
    image = crop(image)
    image.save(filename+'.png')
def draw_arrow_right(filename):
    image = draw_arrow()
    image = image.rotate(90)
    image = crop(image)
    # 保存图像
    image.save(filename+'.png')

def draw_black(filename):
        # 创建一张高640宽400的纯黑色的图像
    image = Image.new('RGB', (400, 640), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    # 保存图像
    image.save(filename+'.png')


def draw_one_num(text,filename):
    # 创建一个黑色背景的图像
    image = Image.new('RGB', (400, 640), color='black')

    # 在图像中心创建一个Draw对象
    draw = ImageDraw.Draw(image)
    width, height = image.size
    center = (200,300)

    # 定义要显示的数字

    # 设置字体和字号
    # font = ImageFont.truetype("arial.ttf", 15)
    font = ImageFont.truetype("Keyboard.ttf",150)
    # font =ImageFont.load_default()

    # 获取文本大小并居中

    text_bbox = draw.textbbox(center, str(text), font=font)
    text_position = (center[0] - text_bbox[2] // 2, center[1] - text_bbox[3] // 2)

    # 在图像中心绘制文本
    draw.text((50,200), str(i).zfill(3), font=font, fill='red')
    # 显示图像
    image.save(filename+'.png')


def get_png(arr):

    os.mkdir("arrow_pic")
    root = os.path.join("arrow_pic","C_")
    for idx,item in enumerate(arr):
        if item ==0:
            draw_black(root + str(idx))
        if item ==1:
            draw_arrow_up(root + str(idx))
        if item ==2:
            draw_arrow_down(root + str(idx))
        if item ==3:
            draw_arrow_left(root + str(idx))
        if item == 4:
            draw_arrow_right(root + str(idx))
    idx +=1
    for i in range(3):
        draw_black(root + str(idx + i))


def draw_num(i):
    save_dir = "arrow_num"
    os.mkdir(save_dir)
    root = os.path.join(save_dir, "C_")
    for idx in range(i):
        draw_one_num(idx,root + str(idx).zfill(3))

# get_png([1,2,3,4])
# draw_num(33)

# draw_num(512)
draw_num("start")