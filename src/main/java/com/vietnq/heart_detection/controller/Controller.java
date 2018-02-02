package com.vietnq.heart_detection.controller;

import com.google.gson.Gson;
import com.vietnq.heart_detection.Utils;
import com.vietnq.heart_detection.model.Classifier;
import com.vietnq.heart_detection.model.RectF;
import com.vietnq.heart_detection.model.TensorFlowObjectDetectionAPIModel;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@RestController
public class Controller implements InitializingBean {
    public static final String MODEL_NAME = "ssd_mobilenet_v1_android_export2.pb";
    public static final String LABEL_NAME = "coco_labels_list2.txt";
    public static final int INPUT_SIZE = 300;

    private String resourcePath = getClass().getClassLoader().getResource("").getPath() + "static/";
    private byte[] model;
    private List<String> label = new ArrayList<>();
    private TensorFlowObjectDetectionAPIModel apiModel;

    @PostMapping(value = "/tensorflow")
    public String tensorflow(@RequestPart("image") MultipartFile multipartFile) throws Exception {
        byte originalImageByte[] = multipartFile.getBytes();
        BufferedImage original = null;
        if (multipartFile.getContentType().contains("image")) {
            original = ImageIO.read(new ByteArrayInputStream(originalImageByte));
        }
        if (original == null) {
            return null;
        }
        BufferedImage resized = Utils.resizeImage(original, original.getType());

        byte resizedImageByte[] = ((DataBufferByte) resized.getRaster().getDataBuffer()).getData();

        long start = System.currentTimeMillis();
        List<Classifier.Recognition> results = apiModel.recognizeImage(resizedImageByte);


        for (int i = 0; i < results.size(); i++) {
            Classifier.Recognition result = results.get(i);
            RectF location = result.getLocation();
            result.setLocation(Utils.resized2original(resized, original, location));
        }
        long end = System.currentTimeMillis() - start;
        System.out.println("Process time : " + end);
        Gson gson = new Gson();
        return gson.toJson(results);
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        Path modelPath = Paths.get(resourcePath, MODEL_NAME);
        Path labelPath = Paths.get(resourcePath, LABEL_NAME);
        System.out.println(modelPath.toString());
        System.out.println(labelPath.toString());
        model = Files.readAllBytes(modelPath);
        label = Files.readAllLines(labelPath);
        System.out.println(model.length + "");
        System.out.println(label.size() + "");
//
//        URL modelUrl = new URL("https://github.com/vietcoscc/TensorflowAPI/raw/master/src/main/resources/static/ssd_mobilenet_v1_android_export.pb");
//        URL labelUrl = new URL("https://raw.githubusercontent.com/vietcoscc/TensorflowAPI/master/src/main/resources/static/coco_labels_list.txt");
//
//        InputStream modelStream = modelUrl.openStream();
//        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
//        StreamUtils.copy(modelStream, outputStream);
//        model = outputStream.toByteArray();
//
//        System.out.println(model.length + "");
//        BufferedReader in = new BufferedReader(
//                new InputStreamReader(labelUrl.openStream()));
//
//        String inputLine;
//        while ((inputLine = in.readLine()) != null) {
//            label.add(inputLine);
//            System.out.println("Line : " + inputLine);
//        }
//        in.close();

        apiModel = TensorFlowObjectDetectionAPIModel.create(model, label, INPUT_SIZE);

    }
}
