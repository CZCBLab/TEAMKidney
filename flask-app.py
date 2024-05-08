import io
import json
import os

from flask import Flask, request
import boto3

from tools.inference import main as inference
from tools.measurement4mask import main as measurement

app = Flask(__name__)

s3 = boto3.resource('s3')


@app.route("/predict", methods=['POST'])
def predict():
    inputfilepath = request.form.get('filename')
    if not inputfilepath:
        return json.dumps({"status": 500, "message": "Bad request - missing filename in request form"})

    try:
        pixelsize = float(request.form.get("pixelsize")) # nm
    except:
        return json.dumps({"status": 500, "message": "Bad request - missing or malformed pixel size in request form"})

    obj = s3.Object(bucket_name=os.environ["AWS_S3_BUCKET_NAME"], key=inputfilepath)
    response = obj.get()
    img = io.BytesIO(response['Body'].read()) # StreamingBody to BytesIO

    try:
        #print("Calling inference...")
        pred, masked = inference(img)
        #pred.show()
        #masked.show()
        #print("Calling measurement...")
        success, msg, measurement_masked_img, fp_widths_json, width_info_list = measurement(img, pred, inputfilepath, pixelsize)
        #measurement_masked_img.show()

        if not success:
            return json.dumps({"status": 400, "message": msg, "srcfile": inputfilepath,})
    except Exception as e:
        # If anything else goes wrong that is not a "no GBM" type error, for example bad filetype
        return json.dumps({"status": 400, "message": "Unable to predict: " + str(e), "srcfile": inputfilepath})

    # The input filepath is expected to look like [userdir]/inputs/[imagefilename].[extension]
    # The output filepath should be [userdir]/[output-type]/[imagefilename].[extension]
    path_segments = inputfilepath.split('/')
    fname_split = path_segments[-1].split('.')
    userdir = path_segments[0]
    fname = '.'.join(fname_split[0:-1])
    input_img_ext = fname_split[-1]
    # output file extension is currently hardcoded to PNG following Anqi's original code.
    # If someday we want it to mirror the input ext, then:
    # # PIL only understands 'jpeg' not 'jpg'
    # measurement_mask_ext = 'jpeg' if input_img_ext=='jpg' else input_img_ext
    measurement_mask_ext = "png"
    result_mask_filepath = userdir+"/measurementmasks/"+fname+"."+measurement_mask_ext
    result_widths_filepath = userdir+"/widthinfojsons/"+fname+".json"

    with io.BytesIO() as fileobj:
        measurement_masked_img.save(fileobj, format=measurement_mask_ext)
        s3.Bucket(os.environ["AWS_S3_BUCKET_NAME"]).put_object(Key=result_mask_filepath, Body=fileobj.getvalue())

    assert len(width_info_list) == 1 and len(fp_widths_json) == 1
    width_info = width_info_list[0]
    # Just combine the FP widths list into the width info JSON
    width_info['FP_widths'] = fp_widths_json[0]['FP_widths']

    width_info_json = json.dumps({
        "image_id": fname,
        # cast from np types to python types...
        "skeleton_length": int(width_info['skeleton_length']),
        "area": float(width_info['area']),
        "GBM_mean_width": float(width_info['GBM_mean_width']),
        "FP_num": int(width_info['FP_num']),
        "FP_mean_width": float(width_info['FP_mean_width']),
        "FP_widths": width_info['FP_widths'],
        "pixel_size": pixelsize,
        "unit": "nm", # All (numerical, non-enumerative) fields in this dict are expressed in this unit.

        # Metadata: Hrm. I avoided putting metadata in the width_info_json for as long as possible.
        # But if we want to (1) support non-png input images, and someday maybe also support non-png
        # output images; and (2) enable the user to click through files in the web client and
        # view old inputs and results, then this info has to be stored somewhere.
        # If the users do not like this info being in the "science results" JSON, then we can
        # move it out to a separate userdir+"/metadatajsons/"+fname+".json" and do the necessary
        # in the web client. But for now, just keeping it simple.
        "input_img_ext": input_img_ext,
        "measurement_mask_ext": measurement_mask_ext,
    })
    s3.Bucket(os.environ["AWS_S3_BUCKET_NAME"]).put_object(Key=result_widths_filepath, Body=width_info_json)

    response_json = json.dumps({
        "status": 200,
        "message": msg,
        "srcfile": inputfilepath,
    })
    return response_json
