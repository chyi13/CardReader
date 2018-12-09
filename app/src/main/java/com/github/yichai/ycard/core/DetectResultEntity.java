package com.github.yichai.ycard.core;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.Comparator;

public class DetectResultEntity {
    public Rect boundingBox;
    public Mat mat;
    String result;

    public DetectResultEntity(Rect boundingBox, Mat mat, String result) {
        this.boundingBox = boundingBox;
        this.mat = mat;
        this.result = result;
    }

    public static class DetectResultEntityComp implements Comparator<DetectResultEntity> {
        // Used for sorting in ascending order of
        // roll number
        public int compare(DetectResultEntity a, DetectResultEntity b)
        {
            return a.boundingBox.x - b.boundingBox.x;
        }
    }
}
