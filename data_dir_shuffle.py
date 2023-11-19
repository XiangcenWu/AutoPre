import glob
import random



    
def read_data_list(data_list_txt):
    l = []
    with open(data_list_txt, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            l.append(x)
    return l


if __name__ == "__main__":
    
    dir_list = sorted(glob.glob('/home/xiangcen/AutoPre/MSD_prostate_h5/*.h5'))
    random.shuffle(dir_list)
    with open(r'/home/xiangcen/AutoPre/MSD_prostate_h5.txt', 'w') as fp:
        for item in dir_list:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    
    l = read_data_list('/home/xiangcen/AutoPre/MSD_prostate_h5.txt')
    # display list
    print("num of data:", len(l))