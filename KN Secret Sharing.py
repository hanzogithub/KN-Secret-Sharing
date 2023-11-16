!pip install opencv-python-headless
import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow

N = input('Enter the total number of shares (N) : ')
N = int(N)
print("The total number of shares (N) = ", N)
K = input("Enter the number of shares required to retreive the secret image (K) :")
K = int(K)
print("The number of shares required to retreive the secret image (K) = ", K)

# Function to read the secret image
def read_image(img_name):
    img = cv.imread(img_name)
    resized_img = cv.resize(img, (500,500))
    cv.waitKey(0)
    return img

image_name = input("Enter the name of the image (with extension)")
image = read_image(image_name)
print("The shape of the secret image is = ", image.shape)
cv2_imshow(image)

# Function to generate a random key
def generateRandomKey(key_shape):
    dealer_key = np.zeros(key_shape).astype(int)
    for i in range(0, len(dealer_key)):
        for j in range(0, len(dealer_key[i])):
            for k in range(0, 3):
                dealer_key[i][j][k] = (np.random.randint(0,255))
    return dealer_key

random_key = generateRandomKey(image.shape)
print("The shape of random key = ", random_key.shape)

# Function to convert the secret image into encrypted image
def convertToEncryptedImage(im, rand_key):
    print("Converting to Encrypted Image...")
    encrypted_im = np.zeros(im.shape).astype(int)

    for i in range(0, len(rand_key)):
        for j in range(0, len(rand_key[i])):
            for k in range(0, 3):
                 encrypted_im[i][j][k] = im[i][j][k]^rand_key[i][j][k]
    cv2_imshow(encrypted_im)
    return encrypted_im

encrypted_image = convertToEncryptedImage(image, random_key)
print("The shape of encrypted image = ", encrypted_image.shape)

# Function to generate unique id's for each participant
def getUniqueIds(n):
    temp_arr = np.zeros(n).astype(int)
    for i in range(n):
        temp_arr[i] = np.random.randint(0,255)
    return temp_arr

unique_ids = getUniqueIds(N)
print("The unique id's of all the N participants are :")
print(unique_ids)

#Function to get encrypted id's for each participant
def getEncryptedIds(unique_ids, n):
    temp_arr = []
    for i in range(n):
        x = np.zeros(8).astype(int)
        s = '{0:08b}'.format(int(unique_ids[i]))
        msb = s[0:4]
        lsb = s[4:8]
        for i in range(4):
            x[i] = int(msb[i])
        for i in range(4):
            x[i+4] = int(msb[i])^int(lsb[i])
        res = int("".join(str(l) for l in x), 2)
        temp_arr.append(res)
    return temp_arr
encrypted_ids = getEncryptedIds(unique_ids, N)

# Function to get authenticated image for each participant (R1)
def getR1(en_image, sh):
    R1 = np.zeros(en_image.shape).astype(int)
    R1_remainder = np.zeros(en_image.shape).astype(int)
    for i in range(0, len(R1)):
        for j in range(0, len(R1[i])):
            for k in range(0, 3):
                R1[i][j][k] = int(en_image[i][j][k]//int(sh))
                R1_remainder[i][j][k] = en_image[i][j][k]%int(sh)
    return [R1, R1_remainder]

authenticated_image = getR1(encrypted_image, K) # array of R1 and R1_remainder

# Left circular rotation of an array str -> xa times
def leftRotate(bit_8_s, xa):
    temp_str = []
    for i in range(xa):
        temp_str.append(bit_8_s[i])
    check_point = 0
    for i in range(0, len(bit_8_s)-xa):
        bit_8_s[i] = bit_8_s[i+xa]
        check_point = i
    for i in range(0,xa):
        bit_8_s[check_point+1+i] = temp_str[i]
    return bit_8_s

#Function to perform left circular shift
def leftCircularShift(R_x, e_x):
    temporary_R1 = np.zeros(R_x.shape).astype(int)
    for i in range(0, len(temporary_R1)):
        for j in range(0, len(temporary_R1[i])):
            for k in range(0, 3):
                s = '{0:08b}'.format(R_x[i][j][k])
                tempo = np.zeros(8).astype(int)
                for l in range(8):
                    tempo[l] = int(s[l])
                e_enx = e_x%8
                tempo = leftRotate(tempo, e_enx)
                res = int("".join(str(l) for l in tempo), 2)
                temporary_R1[i][j][k] = res
    return temporary_R1

#Generating N Shares
def generate_N_shares(Rg, n, n_ids):
    S = []
    for i in range(0,n):
        temp_g = Rg
        tempo_S = leftCircularShift(temp_g, n_ids[i])
        S.append(tempo_S)
    return S

h1 = authenticated_image[0]
shares = generate_N_shares(h1, N, encrypted_ids)

#Saving N Shares
def saveNshares(S, k):
    for i in range(0, k):
        filename = "share_"+str(i+1)+".png"
        if S[i] is not None:
          cv.imwrite(filename, S[i])
        if S[i] is not None:
          print("The " + str(i + 1) + " th share is as follows : ")
        if S[i] is not None:
          cv2_imshow(S[i])

saveNshares(shares, N)


# Function to encode the 8 bit encrypted ids into each of the corresponding shares
def encode_shares(S, n, encrypted_ids):
    for l in range(0,n):
        encrypted_string = '{0:08b}'.format(encrypted_ids[l])
        count_bits = 0
        temp_encoded = np.zeros(S[l].shape).astype(int)
        for i in range(0, len(S[l])):
            for j in range(0, len(S[l][i])):
                for k in range(0, 3):
                    temp_string = '{0:08b}'.format(S[l][i][j][k])
                    temp_array = np.zeros(8).astype(int)
                    for b in range(8):
                        temp_array[b] = int(temp_string[b])
                    temp_array[7] = encrypted_string[count_bits]
                    res = int("".join(str(l) for l in temp_string), 2)
                    temp_encoded[i][j][k] = res
                    count_bits+=1
                    if(count_bits>=8):
                        break
                if(count_bits>=8):
                        break
            if(count_bits>=8):
                        break
    return S

# Right circular rotation of an array str -> xa times
def rightRotate(bit_8_s, xa):
    temp_str = []
    for i in range(len(bit_8_s)-xa,len(bit_8_s)):
        temp_str.append(bit_8_s[i])
    for i in range(len(bit_8_s)-1, xa-1, -1):
        bit_8_s[i] = bit_8_s[i-xa]
    for i in range(0,xa):
        bit_8_s[i] = temp_str[i]
    return bit_8_s

#Function to perform right circular shift
def rightCircularShift(R_x, e_x):
    tempo_R_x = np.zeros(R_x.shape).astype(int)
    for i in range(0, len(tempo_R_x)):
        for j in range(0, len(tempo_R_x[i])):
            for k in range(0, 3):
                s = '{0:08b}'.format(R_x[i][j][k])
                tempo = np.zeros(8).astype(int)
                for l in range(8):
                    tempo[l] = int(s[l])
                e_x_r = e_x%8
                tempo_r = rightRotate(tempo, e_x_r)
                res = int("".join(str(l) for l in tempo_r), 2)
                tempo_R_x[i][j][k] = res
    return tempo_R_x

# Retreiving shares
def retreive_shares(S, k, encrypted_ids):
    R = []
    for i in range(0,k):
        temp = S[i]
        temp_S = rightCircularShift(temp, encrypted_ids[i])
        R.append(temp_S)
    return R

#user will input k unique ids and their corresponding shares
id_list = []
retrieved_shares = []

for i in range(K):
   id = input("Enter unique ID: ")
   img_name = input("Enter corresponding share name(with extension) : ")
   image = read_image(img_name)
   id_list.append(id)
   retrieved_shares.append(image)

encrypted_ids = getEncryptedIds(id_list, K)
encoded_shares = encode_shares(retrieved_shares, K, encrypted_ids)
retreived_shares = retreive_shares(encoded_shares, K, encrypted_ids)

# Retreive Secret Image
def getSecretImage(retreived_shares, k, r_shape, rem, key, org_img, org_share, final_image):
    ret = np.zeros(r_shape).astype(int)
    for l in range(k):
        for i in range(len(retreived_shares[l])):
            for j in range(len(retreived_shares[l][i])):
                for t in range(3):
                    ret[i][j][t] += retreived_shares[l][i][j][t]

    for i in range(len(ret)):
            for j in range(len(ret[i])):
                for t in range(3):
                    ret[i][j][t] = ret[i][j][t] + rem[i][j][t]

    secret_image = np.zeros(r_shape).astype(int)
    for i in range(len(ret)):
            for j in range(len(ret[i])):
                for t in range(3):
                    temp_xor = ret[i][j][t]^key[i][j][t]
                    secret_image[i][j][t] = temp_xor

    fileName = "retreived_image.png"
    print("The retreived image is as follows : ")
    cv2_imshow(secret_image)
    cv.imwrite(fileName, secret_image)

getSecretImage(retreived_shares, K, image.shape, authenticated_image[1], random_key, encrypted_image, authenticated_image[0], image)