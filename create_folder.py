import os
import shutil
import random



def main(path, out, num):
    files = os.listdir(path)
    count = 0
    while count <= num:
        count += 1
        index = random.randint(0, len(files)-1)
        file = files[index]
    # for files in os.listdir(path):
        name = os.path.join(path, file)
        back_name = os.path.join(out, file)
        if os.path.isfile(name):
            shutil.copy(name, back_name)
        else:
            if not os.path.isdir(back_name):
                os.makedirs(back_name)
            main(name, back_name)

 
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                 
		os.makedirs(path)     
 
	# else:
	# 	print("There is this folder!")

    # return path


if __name__ == '__main__':
    A = "train"
    B = "testing_train"
    files = os.listdir(A)
    for file in files:
        new_path = os.path.join(B, file)
        mkdir(new_path)

        main(os.path.join(A, file), new_path, 20)
