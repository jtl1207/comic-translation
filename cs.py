from PIL import Image, ImageDraw, ImageFont

# create an image
out = Image.new("RGB", (500, 300), (255, 255, 255))

# get a drawing context
d = ImageDraw.Draw(out)

# draw multiline text
d.text(xy=(20, 20),
       text="你那还好吗?!,",
       direction='ttb',
       font=ImageFont.truetype("covermaker/fonts/华康翩翩体简-粗体.otf", 20),
       fill=(0, 0, 0),
       anchor='mt',
       spacing=30,
       align='center',
       stroke_width=2,
       stroke_fill="#0f0")


out.show()
