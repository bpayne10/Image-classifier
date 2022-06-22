def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    cuda = torch.cuda.is_available
    if cuda:
        model.cuda()
        print("Number of GPUs: ", torch.cuda.device_count())
        print("Devide Name: " , torch.cuda.get_device_name(torch.cuda.device_count())-1)
    else:
        model.cpu()
        print("CPU is used")
        
    model.eval()
     
    image = process_image(image_path)
    image = Variable(image)
    if cuda:
        image = image.cuda
    output = model.forward(image)    
    probabilities = torch.exp(output).data
    
    prob = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind =[]
    for i in range(len(model.class_to_idx.items()))
        ind.append(list(model.class_to_idx.items))[i][0]
    
    label = []
    for i in range(5):
        label.append(ind[index[i]])
        
    return prob, label

# TODO: Display an image along with the top 5 classes
prob, classes = predict(img_path, model)
max_index = np.argmax(prob)
max_probability = prob[max_index]
label = classes[max_index]

fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

image = image.open(img_path)
ax1.axis('off')
ax1.set_titel(cat_to_name[label])
ax1.imshow(image)

labels = []
for cl in classes:
    labels.append(cat_to_name[cl])
    
y_pos = np.array(5)
ax2.set_ytricks(y_pos)
ax2.set_ytricklabels(labels)
ax2.set_xlabel('Probability')
ax2.invert_yaxis()
ax2.barh(y_pos, prob, xerr=0, align='center', color='blue')

plt.show()