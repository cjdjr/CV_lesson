import random
import shutil

if __name__=="__main__":
    random.seed(19260817)
    for type in range(1,18):
        index=[i for i in range(1,81)]
        random.shuffle(index)
        base=(type-1)*80
        for i in range(60):
            name="image_"+str(base+index[i]).zfill(4)
            shutil.copy("jpg/"+name+".jpg", "train_set/")
            with open("train_set/"+name+".txt", 'w') as f:
                f.write(str(type))

        for i in range(60,80):
            name = "image_" + str(base + index[i]).zfill(4)
            shutil.copy("jpg/" + name + ".jpg", "test_set/")
            with open("test_set/"+name+".txt", 'w') as f:
                f.write(str(type))
    