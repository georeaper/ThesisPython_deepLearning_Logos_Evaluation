from PIL import Image
from resizeimage import resizeimage
diri="D:/pythonprogs/Lungs/test/images/"
i="00000001_000.png "
dir_dest="D:/pythonprogs/Lungs/test/image_new_dim/"
im= Image.open(diri+i)
cover=resizeimage.resize_cover(im,[225,225])
cover.save(dir_dest+i,im.format)