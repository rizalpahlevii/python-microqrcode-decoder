import numpy as np
import pyboof as pb
from flask import Flask, render_template, Response
from fastapi import FastAPI
# Detects all the QR Codes in the image and prints their message and location
data_path = "sample/tbanda.jpeg"

detector = pb.FactoryFiducial(np.uint8).microqr()

image = pb.load_single_band(data_path, np.uint8)

detector.detect(image)

print("Detected a total of {} Micro QR Codes".format(len(detector.detections)))

for qr in detector.detections:
    print("Message: " + qr.message)
    print("     at: " + str(qr.bounds))