#Mark Joseph
import numpy as np
import re
from scipy import signal
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import cmath
import matplotlib.image as mpimg




#parameters for morlet functions
sigmas = [1, 3, 6]
thetas = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi*2/3, np.pi*3/4, np.pi*5/6]

# we need to plot two graphs, a real and an imiginary. the morlet function returns a complex number so we need to extract the real and imiginary
# from that function thus the morlet real and imiginary. in order to do that function we need c1 and c2 then we can do that function for each pixel
# the morlet function is a complex function that taat returns a complex number,
# it takes a complex number which is rewritten from e^i etc to cos(something) + i sin(something)

#define the morlet function that return the real part
def morlet_real(x, y, sig, theta, C1, C2):
    # set variables
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    pie = np.pi
    # one peak morlet wave function with greek letter equal to 4
    exponentOfEInsideBrackets = (pie / (2 * sig)) * ((x * cosTheta) + (y * sinTheta))
    exponentOfEOutsideBrackets = -(x**2 + y**2)/ (2 * sig**2)

    #morlet wave function
    #cmath.rect(r, phi) Return the complex number x with polar coordinates r and phi.
    z = C1 / sig * (cmath.rect(1, exponentOfEInsideBrackets) - C2) * np.exp(exponentOfEOutsideBrackets)
    return z.real


#define the morlet function that return the imaginary part
def morlet_imag(x, y, sig, theta, C1, C2):
    # set variables
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    pie = np.pi
    # one peak morlet wave function with greek letter equal to 4
    exponentOfEInsideBrackets = (pie / (2 * sig)) * ((x * cosTheta) + (y * sinTheta))
    exponentOfEOutsideBrackets = -(x**2 + y**2)/ (2 * sig**2)

    #morlet wave function
    #cmath.rect(r, phi) Return the complex number x with polar coordinates r and phi.
    z = C1 / sig * (cmath.rect(1, exponentOfEInsideBrackets) - C2) * np.exp(exponentOfEOutsideBrackets)
    return z.imag


#finds the constants c2
def find_c2(xymin, xymax, sig, theta):
    numerator = 0
    denominator = 0
    cosine = np.cos
    cosineTheta = np.cos(theta)
    sineTheta = np.sin(theta)
    pie = np.pi
    for x in range(xymin, xymax+1, 1):
        for y in range( xymin, xymax+1, 1):
            numerator = numerator + (cosine((pie / (2 * sig)) * ((x * cosineTheta) + (y * sineTheta))) * np.exp(-(x**2 + y**2)/(2 * sig**2)))
            denominator = denominator + (np.exp(-(x**2 + y**2)/(2 * sig**2)))

    C2 = numerator/denominator
    return C2


#finds the constant c1
def find_c1(xymin, xymax, sig, theta, C2):
    Z = 0
    pie = np.pi
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    cosine = np.cos

    for x in range(xymin, xymax+1, 1):
        for y in range( xymin, xymax+1, 1):
            Z = Z + (1 - 2* C2 * cosine(pie/(2*sig) * ((x * cosTheta) + (y * sinTheta))) + C2**2) * np.exp((-(x**2 + y**2)/sig**2))
    C1 = 1/np.sqrt(Z)

    return C1


#plot the morlet function for the real
def morletMatrix_real(xymin, xymax, sig, theta):

    #find c1 and c2
    C2 = find_c2(xymin, xymax, sig, theta)
    C1 = find_c1(xymin, xymax, sig, theta, C2)

    #define grid over which the function should be plotted
    xx, yy = np.meshgrid(np.linspace(xymin, xymax, 33),np.linspace(xymin, xymax, 33))

    # fill a matrix with the morlet function values
    zz= np.zeros(xx.shape)
    for i in range(yy.shape[0]):
        for j in range(xx.shape[0]):
            zz[i,j] = morlet_real(xx[i,j], yy[i,j], sig, theta, C1, C2)

    return zz

# plot morlet function for imiginary
def morletMatrix_imag(xymin, xymax, sig, theta):
    #determine constants
    C2 = find_c2(xymin, xymax, sig, theta)
    C1 = find_c1(xymin, xymax, sig, theta, C2)

    #define grid over which the function should be plotted
    xx = np.meshgrid(np.linspace(xymin, xymax, xymax-xymin+1))
    yy = np.meshgrid(np.linspace(xymin, xymax, xymax-xymin+1))

    # fill a matrix with the morlet function values
    zz= np.zeros(xx.shape)
    for i in range(yy.shape[0]):
        for j in range(xx.shape[0]):
            zz[i,j] = morlet_imag(xx[i,j], yy[i,j], sig, theta, C1, C2)

    return zz

# genetate left and right convolved piano images
#==============================================================================================

leftImage= imread("im0.png", True)
rightImage= imread("im1.png", True)


kernel = morletMatrix_real(-16, 16, 6, 0)
leftZeroWaveletResponse = signal.convolve2d(leftImage, kernel, boundary='symm', mode='same')
# plt.figure()
# plt.imshow(np.absolute(leftZeroWaveletResponse), cmap='gray')
# plt.savefig("leftZero")

kernel = morletMatrix_real(-16, 16, 6, np.pi / 2)
leftPiTwoWaveletResponse = signal.convolve2d(leftImage, kernel, boundary='symm', mode='same')
# plt.figure()
# plt.imshow(np.absolute(leftPiTwoWaveletResponse), cmap='gray')
# plt.savefig("leftpi")

kernel = morletMatrix_real(-16, 16, 6, 0)
rightZeroWaveletResponse = signal.convolve2d(leftImage, kernel, boundary='symm', mode='same')
# plt.figure()
# plt.imshow(np.absolute(rightZeroWaveletResponse), cmap='gray')
# plt.savefig("rightZero")

kernel = morletMatrix_real(-16, 16, 6, np.pi / 2)
rightPiTwoWaveletResponse = signal.convolve2d(leftImage, kernel, boundary='symm', mode='same')
# plt.figure()
# plt.imshow(np.absolute(rightPiTwoWaveletResponse), cmap='gray')
# plt.savefig("rightpi")

print ("We now have the convolved parts")

# get the occlusion
# ================================================================================================================

def load_pfm(file):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().rstrip()
  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.reshape(data, shape), scale

# open file
leftFile = open("disp0.pfm", "r")
rightFile = open("disp1.pfm", "r")

# load the pfm files
leftImage = load_pfm(leftFile)
rightImage = load_pfm(rightFile)

leftImage = leftImage[0]
rightImage = rightImage[0]

left = plt.figure(1)
ax = plt.imshow(leftImage, origin='lower')
left.show()

right = plt.figure(2)
plt.imshow(rightImage, origin='lower')
right.show()


occlusion = np.copy(leftImage)

for i in range(1920):
    for j in range(2820):
        if (np.isinf(leftImage[i][j])):

            occlusion[i][j] = 1
        else:
            occlusion[i][j] = 0

occlusionL = plt.figure(3)
plt.imshow(occlusion, origin='lower')
occlusionL.show()
plt.savefig("occlusion")

print("\nWe now have the occlusions")

# #===============================================================
# #create an n x m matrix to store matching values
# #need to round values to integers
width =2820
height = 1920
matchingArray=[[0]*width for i in range(height)]# used to store matching pixels
best = []# used to store the best disparaty match
l = []#used to store places that have tension
# min and max values from the left disparaty array
dmin = np.amin(leftImage)
dmax = np.nanmax(leftImage)


tolerance = 0
# we use a range starting at 7 because the convolved image used a scale of 6 so pixels
# outside of this filter will be irrelevant
print("\nStarting disparaty stuff")
for i in range(7,width-6):
  # count is used to count how many pixels passed the test
  count =0
  for j in range(7, height-6):
    # we only want to deal with pixels that are not occluded(occlusion is 0)
    if(occlusion[i][j] == 0):
        leftResponseAtZero = 0
        leftResponseAtPie = 0
        # if the wavelet response is above the tolerance, meaning we have contrast, we do something(we only work with pixels that have tension)
        # we use the 0 and pi/2 angles, and once there is one angle that passes, we work
        if (leftPiTwoWaveletResponse[i][j] > tolerance):
            leftResponseAtZero = leftPiTwoWaveletResponse[i][j]
        if(leftZeroWaveletResponse[i][j] > tolerance):
            leftResponseAtPie = leftZeroWaveletResponse[i][j]

        tempMinDiff = 10000000
        # min and max are from the left disparaty image
        #loop from the min to the max and check each right pixel
        for w in range(dmin,dmax):
            # for each posible left disparaty/match, we get the right
            r = occlusion[i][j] - w
            # if r is within the range
            rightResponseAtPie = 0
            rightResponseAtZero = 0
            if(7<= r <= height-7):
                # save the values
                if(rightPiTwoWaveletResponse[i][j] > tolerance):
                    rightResponseAtPie = rightPiTwoWaveletResponse[i][j]
                if(rightZeroWaveletResponse[i][j] > tolerance):
                    rightResponseAtZero = rightZeroWaveletResponse[i][j]

                # we need to compute the difference for matching values of the angles,
                # if they are a good match, then they will have a small difference
                if(leftResponseAtPie > 0 and rightResponseAtPie > 0):
                    difference = rightResponseAtPie - leftResponseAtPie
                    #the difference will be a complex number so we need to make 1 number(complex * conjugate)
                    # fabs is used to get the absolute value
                    diff = math.fabs(math.sqrt(difference * difference.conjugate()))
                    matchingArray[i][j] = diff
                    # we check the best disparaty and store it in an array then update
                    if(diff < tempMinDiff):
                        # first  iteration will have largest value and end up with smalles
                        best.append(w)
                        tempMinDiff =diff
                    count = count + 1
                    # store in a list the places that have tension
                    l.append(w)

                elif(rightResponseAtZero>0 and leftResponseAtPie > 0):
                    difference = rightResponseAtZero - leftResponseAtZero
                    #the difference will be a complex number so we need to make 1 number(complex * conjugate)
                    # fabs is used to get the absolute value
                    diff = math.fabs(math.sqrt(difference * difference.conjugate()))
                    matchingArray[i][j] = diff
                    # we check the best disparaty and store it in an array then update
                    if(diff < tempMinDiff):
                        # first  iteration will have largest value and end up with smalles
                        best.append(w)
                        tempMinDiff =diff
                    count = count + 1
                    # store in a list the places that have tension
                    l.append(w)

                # there was no matching values for the angles so we continue
                else:
                    continue
            # r was not in the range so try other value
            else:
                continue

    # neither angle, 0 or pi/2 passed the tolerance therefore we continue
    else:
        continue
print("\nProgram finished")






