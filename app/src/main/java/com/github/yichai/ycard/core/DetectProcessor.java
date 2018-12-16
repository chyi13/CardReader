package com.github.yichai.ycard.core;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.widget.Toast;

import com.github.yichai.ycard.MainActivity;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;

public class DetectProcessor {

    private static boolean caputured = false;

    private static final boolean DEBUG = false;

    private static Context context;

    public static Mat process(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat rgba = inputFrame.rgba();
        // convert to gray
        Mat gray = inputFrame.gray();
        Mat mIntermediateMat = new Mat();
        // gaussian blur
        Imgproc.GaussianBlur(gray, mIntermediateMat, new Size(5, 5), 0);
        // canny edge
        Imgproc.Canny(mIntermediateMat, mIntermediateMat, 80, 100);
        // find contours
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat contoursMat = new Mat();
        Imgproc.findContours(mIntermediateMat, contours, contoursMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        MatOfPoint2f approxCurve = new MatOfPoint2f();

        for (MatOfPoint cnt : contours) {

            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());

            double contourArea = Imgproc.contourArea(cnt);
            if (Math.abs(contourArea) < 10000) {
                continue;
            }
            Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);
            int numberVertices = (int) approxCurve.total();
            if (numberVertices == 4) {
                Point p0 = new Point(approxCurve.get(0, 0));
                Point p1 = new Point(approxCurve.get(1, 0));
                Point p2 = new Point(approxCurve.get(2, 0));
                Point p3 = new Point(approxCurve.get(3, 0));

                Rect boundingBox = Imgproc.boundingRect(cnt);

                p0.x -= boundingBox.tl().x;
                p1.x -= boundingBox.tl().x;
                p2.x -= boundingBox.tl().x;
                p3.x -= boundingBox.tl().x;

                p0.y -= boundingBox.tl().y;
                p1.y -= boundingBox.tl().y;
                p2.y -= boundingBox.tl().y;
                p3.y -= boundingBox.tl().y;

                RotatedRect minAreaRect = Imgproc.minAreaRect(curve);

                List<Point> source = new ArrayList<Point>();
                source.add(p1);
                source.add(p2);
                source.add(p3);
                source.add(p0);

                List<Point> reorderedList = reorder(source);

                Mat startM = Converters.vector_Point2f_to_Mat(reorderedList);

                Mat contourMat = new Mat(gray, boundingBox);

                Mat wrapM = warp(contourMat, startM, minAreaRect.boundingRect());
                Mat resizeM = new Mat();
                Imgproc.resize(wrapM, resizeM, new Size(500, 315));
//                Mat targetMat = new Mat(resizeM, new Range(250, 300), new Range(170, 450));


                if (!caputured) {
                    Mat result = findTextarea(resizeM);
//                    caputured = true;
                }
            }
        }

        return rgba;
    }

    private static List<Point> reorder(List<Point> points) {
        // find top left => min sum
        Point tl = points.get(0);
        for (int i = 1; i< 4; i++) {
            if ((tl.x + tl.y) > (points.get(i).x + points.get(i).y)) {
                tl = points.get(i);
            }
        }

        // find bottom right => max sum
        Point br = points.get(0);
        for (int i = 1; i< 4; i++) {
            if ((br.x + br.y) < (points.get(i).x + points.get(i).y)) {
                br = points.get(i);
            }
        }

        // find top right => max x - y
        Point tr = points.get(0);
        for (int i = 1; i< 4; i++) {
            if ((tr.x - tr.y) < (points.get(i).x - points.get(i).y)) {
                tr = points.get(i);
            }
        }

        // find bottom left => min x - y
        Point bl = points.get(0);
        for (int i = 1; i< 4; i++) {
            if ((bl.x - bl.y) > (points.get(i).x - points.get(i).y)) {
                bl = points.get(i);
            }
        }

        List<Point> reorderList = new ArrayList<>();
        reorderList.add(tl);
        reorderList.add(tr);
        reorderList.add(br);
        reorderList.add(bl);

        return reorderList;
    }

    private static Mat warp(Mat inputMat, Mat startM, Rect rect) {
        int resultWidth = rect.width;
        int resultHeight = rect.height;

        Mat outputMat = new Mat(resultWidth, resultHeight, CvType.CV_8UC1);

        Point ocvPOut1 = new Point(0, 0);
        Point ocvPOut2 = new Point(resultWidth, 0);
        Point ocvPOut3 = new Point(resultWidth, resultHeight);
        Point ocvPOut4 = new Point(0, resultHeight);

        List<Point> dest = new ArrayList<Point>();
        dest.add(ocvPOut1);
        dest.add(ocvPOut2);
        dest.add(ocvPOut3);
        dest.add(ocvPOut4);
        Mat endM = Converters.vector_Point2f_to_Mat(dest);
        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(startM, endM);
        Imgproc.warpPerspective(inputMat,
                outputMat,
                perspectiveTransform,
                new Size(resultWidth, resultHeight), Imgproc.INTER_CUBIC);
        return outputMat;
    }

    private static void drawCardBound(Mat paint, MatOfPoint2f approxCurve) {

        for (int i = 0; i< 4; i++) {
            Imgproc.circle(paint, new Point(approxCurve.get(i, 0)), 10, new Scalar(255, 0, 0));
            Imgproc.putText(paint, "P" + i, new Point(approxCurve.get(i, 0)), FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 0, 255));
            Imgproc.line(paint, new Point(approxCurve.get(i, 0)), new Point(approxCurve.get((i + 1) % 4, 0)), new Scalar(0, 255, 0), 10);
        }
    }

    private static Mat findTextarea(Mat input) {
        // original
        drawResult(input, 0);
        // gaussian blur https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
        Mat gaussian = new Mat();
        Imgproc.bilateralFilter(input, gaussian,9,90,90);
        drawResult(gaussian, 1);
        // binary
        Mat threshMat = new Mat();
        Imgproc.adaptiveThreshold(gaussian, threshMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 2);
        drawResult(threshMat, 2);
        // erode
        Mat erodeMat = new Mat();
        double erodeKernelSize = 0.9;
        Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new Size(2 * erodeKernelSize + 1, 2 * erodeKernelSize + 1),
                new Point(erodeKernelSize, erodeKernelSize));
        Imgproc.erode(threshMat, erodeMat, erodeElement);
        drawResult(erodeMat, 3);

        // dilate
        Mat dilateMat = new Mat();
        double kernelSize = 0.8;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));
        Imgproc.dilate(erodeMat, dilateMat, element);
        for (int i = 0; i< 9; i++) {
            Imgproc.dilate(dilateMat, dilateMat, element);
        }
        drawResult(dilateMat, 4);
        // find sub contours
        List<MatOfPoint> subContours = new ArrayList<MatOfPoint>();
        Mat subContoursMat = new Mat();
        Imgproc.findContours(dilateMat, subContours, subContoursMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        int j = 0;
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        for (MatOfPoint subCnt : subContours) {
            MatOfPoint2f curve = new MatOfPoint2f(subCnt.toArray());

            double contourArea = Imgproc.contourArea(subCnt);
            if (Math.abs(contourArea) < 500) {
                continue;
            }
            Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);
            int numberVertices = (int) approxCurve.total();

            Rect boundingBox = Imgproc.boundingRect(subCnt);
            System.out.println("width" + boundingBox.width + " height " + boundingBox.height);
            if (boundingBox.width / boundingBox.height > 4 && boundingBox.width > 230) {
//                Mat tempResult = getPerspective(input, curve, approxCurve);
//                drawResult(tempResult, 5);
                Mat contourMat = new Mat(input, boundingBox);
                findDigits(contourMat);
                Imgproc.rectangle(input, boundingBox.tl(), boundingBox.br(), new Scalar(255, 100, 0), 1);
            }
            j++;
        }
        drawResult(input, 6);
        return input;
    }

    private static void findDigits(Mat input) {
        // gaussian blur https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
        Mat gaussian = new Mat();
        Imgproc.bilateralFilter(input, gaussian,9,90,90);
        drawResult(gaussian, 1);
        // binary
        Mat threshMat = new Mat();
        Imgproc.adaptiveThreshold(gaussian, threshMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 2);
        drawResult(threshMat, 2);

        Mat result = new Mat();

        List<MatOfPoint> subContours = new ArrayList<>();
        drawResult(input, 8);
        Imgproc.findContours(threshMat, subContours, result, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        int count = 0;

        List<DetectResultEntity> digits = new ArrayList<>();

        int validContourCount = 0;

        for (MatOfPoint subCnt : subContours) {
            MatOfPoint2f curve = new MatOfPoint2f(subCnt.toArray());

            double contourArea = Imgproc.contourArea(subCnt);
            System.out.println("contourArea " + contourArea);
            if (Math.abs(contourArea) > 30) {
                validContourCount ++;
            }
        }

        if (validContourCount != 18) {
            return;
        } else {
            caputured = true;
        }

        for (MatOfPoint subCnt : subContours) {
            MatOfPoint2f curve = new MatOfPoint2f(subCnt.toArray());

            double contourArea = Imgproc.contourArea(subCnt);
            System.out.println("contourArea " + contourArea);
            if (Math.abs(contourArea) < 30) {
                continue;
            }
            Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);
            int numberVertices = (int) approxCurve.total();

            Rect boundingBox = Imgproc.boundingRect(subCnt);
            System.out.println("width" + boundingBox.width + " height " + boundingBox.height + " left " + boundingBox.x + " y " + boundingBox.y);

            Mat digitMat = new Mat(input, boundingBox);

            Core.bitwise_not(digitMat, digitMat);

//            cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
//                    cv.THRESH_BINARY,11,2)
            Imgproc.adaptiveThreshold(digitMat, digitMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

            Mat enlargeMat = enlargeMat(digitMat);

            String classifyResult = classifyMat(enlargeMat);

            digits.add(new DetectResultEntity(boundingBox, enlargeMat, classifyResult));

            drawResult(enlargeMat, 20 + count);

            count++;

            Imgproc.rectangle(input, boundingBox.tl(), boundingBox.br(), new Scalar(255, 100, 0), 1);
            drawResult(input, 7);
        }

        Collections.sort(digits, new DetectResultEntity.DetectResultEntityComp());

        final StringBuilder sortedDigits = new StringBuilder();
        for (DetectResultEntity digit : digits) {
            sortedDigits.append(digit.result);
        }
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(DetectProcessor.context, sortedDigits.toString(), Toast.LENGTH_LONG).show();
            }
        });

    }

    private static String classifyMat(Mat input) {
        Bitmap tempBitmap = convertMatToBitMap(input);
        saveBitmap(tempBitmap, 0);

        return MainActivity.classifier.classifyFrame(tempBitmap);
    }

    private static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        Mat rgb = new Mat();
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_GRAY2RGB);

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            e.printStackTrace();
        }
        return bmp;
    }

    private static Mat enlargeMat(Mat input) {
//        input.copyTo(big_image(cv::Rect(x,y,small_image.cols, small_image.rows)));

        Mat resultMat = new Mat();
        double erodeKernelSize = 0;
        Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new Size(2 * erodeKernelSize + 1, 2 * erodeKernelSize + 1),
                new Point(erodeKernelSize, erodeKernelSize));
        Imgproc.erode(input, resultMat, erodeElement);

        Mat largeMat = new Mat(resultMat.rows() + 10, resultMat.cols() + 10, resultMat.type(), new Scalar(0));

        Mat roi = largeMat.submat(5, 5 + resultMat.rows(), 5, 5 + resultMat.cols());
        resultMat.copyTo(roi);

        Imgproc.resize(largeMat, largeMat, new Size(28, 28));

        return largeMat;
    }

    private static void drawResult(Mat wrap, int index) {
        if (DEBUG) {
            String fileName = Environment.getExternalStorageDirectory().getPath() +
                    "/sample_picture_" + index + ".png";
            System.out.println("fileName " + fileName + " channel " + wrap.channels());
            Imgcodecs.imwrite(fileName, wrap);
        }
    }

    private static void saveBitmap(Bitmap bitmap, int index) {
        String fileName = Environment.getExternalStorageDirectory().getPath() +
                "/bitmap_" + index + ".png";
        try (FileOutputStream out = new FileOutputStream(fileName)) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
            // PNG is a lossless format, the compression factor (100) is ignored
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void setContext(Context context) {
        DetectProcessor.context = context;
    }
}
