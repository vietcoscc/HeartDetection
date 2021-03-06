package com.vietnq.heart_detection.model;


import org.opencv.core.Point;

import java.util.List;

public class Contour {
    private float x;
    private float y;
    private List<Point> arrPoint;

    public Contour() {
    }

    public Contour(float x, float y, List<Point> arrPoint) {
        this.x = x;
        this.y = y;
        this.arrPoint = arrPoint;
    }

    public float getX() {
        return x;
    }

    public void setX(float x) {
        this.x = x;
    }

    public float getY() {
        return y;
    }

    public void setY(float y) {
        this.y = y;
    }

    public List<Point> getArrPoint() {
        return arrPoint;
    }

    public void setArrPoint(List<Point> arrPoint) {
        this.arrPoint = arrPoint;
    }
}
