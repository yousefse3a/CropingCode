import cv2
from PIL import Image
import numpy as np
import os
from pdf2image import convert_from_path
from flask import Flask ,send_file
from flask_restful import Api ,request 
import pandas as pd


app = Flask(__name__)
api = Api(app)

try:
    path = os.path.dirname(os.path.abspath(__file__))
    upload_folder=os.path.join(path.replace("/file_folder",""),"PDF_Folder")
    os.makedirs(upload_folder, exist_ok=True)
    app.config['upload_folder'] = upload_folder
    
    Images_folder=os.path.join(path.replace("/file_folder",""),"Images")
    os.makedirs(Images_folder, exist_ok=True)
    app.config['Images_folder'] = Images_folder

    ImagesAfterCrop_folder=os.path.join(path.replace("/file_folder",""),"ImagesAfterCrop")
    os.makedirs(ImagesAfterCrop_folder, exist_ok=True)
    app.config['ImagesAfterCrop_folder'] = ImagesAfterCrop_folder
    
except Exception as e:
    app.logger.info('An error occurred while creating temp folder')
    app.logger.error('Exception occurred : {}'.format(e))


@app.route('/croping',methods = ['POST'])
def crop():
    try:
        pdf_file = request.files['file']
        pdf_name = pdf_file.filename
        save_path = os.path.join(app.config.get('upload_folder'),pdf_name)
        pdf_file.save(save_path)
        imges=makeImages(save_path,pdf_name)
        return {"data":imges}
    except Exception as e:
        app.logger.info(e)


@app.route('/cancel',methods = ['POST'])
def dele():
    try:
        imgArr = (request.get_json())['imgArr']
        for img in imgArr:
            os.unlink(os.path.join(app.config['ImagesAfterCrop_folder'],f"{img}.jpg"))
        return {"data":"deleted"}
    except Exception as e:
       return {"message":"error occur"} , 400


@app.route('/<img_name>',methods = ['GET'])
def preview(img_name):
    if request.method == 'GET':
        try:
            filename=f"./ImagesAfterCrop/{img_name}"
            return send_file(filename,mimetype='image/jpg')
        except Exception as e:
            app.logger.info(e)



def makeImages(folderPath,pdf_name):

    answers=getAnwers()

    print("Please Wait while the file is being loaded." , pdf_name)
    pdf_name=pdf_name.split('.')[0]
    file = convert_from_path(folderPath)

    for i in range(len(file)):
        # save pdf as jpg
        file[i].save(os.path.join(app.config['Images_folder'],f'{i + 1}.jpg'), "JPEG")
 
    oky = os.listdir(app.config['Images_folder'])

    imagesPath=[]
    for a in oky:
        # Read image:
        inputImage = cv2.imread(os.path.join(app.config['Images_folder'],a))

        # Store a copy for results:
        inputCopy = inputImage.copy()

        # Convert BGR to grayscale:
        grayInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        # Set a lower and upper range for the threshold:
        lowerThresh = 225
        upperThresh = 233

        # Get the lines mask:
        # mask = cv2.inRange(grayInput,36, 70)
        mask = cv2.inRange(grayInput, lowerThresh, upperThresh)

        def areaFilter(minArea, inputImage):
            # Perform an area filter on the binary blobs:
            (
                componentsNumber,
                labeledImage,
                componentStats,
                componentCentroids,
            ) = cv2.connectedComponentsWithStats(inputImage, connectivity=4)

            # Get the indices/labels of the remaining components based on the area stat
            # (skip the background component at index 0)
            remainingComponentLabels = [
                i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea
            ]

            # Filter the labeled pixels based on the remaining labels,
            # assign pixel intensity to 255 (uint8) for the remaining pixels
            filteredImage = np.where(
                np.isin(labeledImage, remainingComponentLabels) == True, 255, 0
            ).astype("uint8")

            return filteredImage

        # Set a filter area on the mask:
        # 50
        minArea = 50
        mask = areaFilter(minArea, mask)
        # Reduce matrix to a n row x 1 columns matrix:
        reducedImage = cv2.reduce(mask, 1, cv2.REDUCE_MAX)
        # Find the big contours/blobs on the filtered image:
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        # Store the lines here:
        separatingLines = []

        # We need some dimensions of the original image:
        imageHeight = inputCopy.shape[0]
        imageWidth = inputCopy.shape[1]

        # Look for the outer bounding boxes:
        for _, c in enumerate(contours):
            # Approximate the contour to a polygon:
            contoursPoly = cv2.approxPolyDP(c, 3, True)

            # Convert the polygon to a bounding rectangle:
            boundRect = cv2.boundingRect(contoursPoly)

            # Get the bounding rect's data:
            [x, y, w, h] = boundRect

            # Start point and end point:
            lineCenter = y + (0.5 * h)
            startPoint = (0, int(lineCenter))
            endPoint = (int(imageWidth), int(lineCenter))

            # Store the end point in list:
            separatingLines.append(endPoint)

            # Draw the line using the start and end points:
            color = (0, 255, 0)
            cv2.line(inputCopy, startPoint, endPoint, color, 2)

            # Show the image:
            # cv2.imshow("",inputCopy)
            image_RGB1 = np.array(inputCopy)
            image1 = Image.fromarray(image_RGB1.astype("uint8")).convert("RGB")
            cv2.waitKey(0)

        # Sort the list based on ascending Y values:
        separatingLines = sorted(separatingLines, key=lambda x: x[1])
        # The past processed vertical coordinate:
        pastY = 0
        # Crop the sections:
        for i in range(len(separatingLines)):
            # Get the current line width and starting y:
            (sectionWidth, sectionHeight) = separatingLines[i]

            # Set the ROI:
            x = 0
            y = pastY
            cropWidth = sectionWidth
            cropHeight = sectionHeight - y

            # Crop the ROI:
            currentCrop = inputImage[y : y + cropHeight, x : x + cropWidth]

            #small arr

            # cv2.imwrite('./ImagesAfterCrop', currentCrop)
            image_RGB = np.array(currentCrop)
            image = Image.fromarray(image_RGB.astype("uint8")).convert("RGB")
            id = int(i) + ((int(a.split('.')[0])-1)*4)
            image = image.save(os.path.join(app.config['ImagesAfterCrop_folder'],f"{pdf_name}-{id}.jpg"))
   

            imgObj={
                'src':(f"http://127.0.0.1:5000/{pdf_name}-{id}.jpg"),
                'answer':(int(answers[id]))
            }

            imagesPath.append(imgObj)
            cv2.waitKey(0)
            # Set the next starting vertical coordinate:
            pastY = sectionHeight
    return imagesPath


def getAnwers():
    try:
        df = pd.read_excel('./answers.xlsx',sheet_name='Sheet1')
        df = df['answer'].to_numpy()
        return df
    except Exception as e:
       return {"message":"error occur"} , 400
    
    

if __name__ == "__main__":
    app.run(debug=True)
