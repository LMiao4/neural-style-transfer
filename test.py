import Style_Transfer

# style images
style_img = image_loader('./style/1.jpg').type(dtype)
# subject images
subject_img = image_loader('./subject/1.jpg').type(dtype)
input_img = subject_img.clone()
assert style_img.size() == subject_img.size(), \
    "we need to import style and content images of the same size"
output = run_style_transfer(cnn, subject_img, style_img, input_img, num_steps=450)
image = output.clone().cpu()
image = image.view(3, imgsize, imgsize)
unloader = transforms.ToPILImage()
image = unloader(image)
image.save('./result1.jpg')
plt.show(image)
plt.title('Result')
