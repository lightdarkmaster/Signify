import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:flutter/material.dart';

class YoloVideo3 extends StatefulWidget {
  const YoloVideo3({super.key});

  @override
  State<YoloVideo3> createState() => _YoloVideo3State();
}

class _YoloVideo3State extends State<YoloVideo3> with WidgetsBindingObserver {
  CameraController? controller;
  late FlutterVision vision;
  List<Map<String, dynamic>> yoloResults = [];
  CameraImage? cameraImage;
  bool isLoaded = false;
  bool isDetecting = false;
  bool _isProcessing = false; // Flag to prevent overlapping processing
  double confidenceThreshold = 0.4;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    initCamera();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Handle app lifecycle changes
    if (controller == null || !controller!.value.isInitialized) return;

    if (state == AppLifecycleState.inactive) {
      stopDetection();
      controller?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      initCamera();
    }
  }

  Future<void> initCamera() async {
    try {
      List<CameraDescription> cameras = await availableCameras();
      if (cameras.isEmpty) {
        debugPrint("No cameras available");
        return;
      }

      vision = FlutterVision();

      // Dispose previous controller if exists
      await controller?.dispose();

      controller = CameraController(cameras[0], ResolutionPreset.high);
      await controller!.initialize();

      await loadYoloModel();
      if (mounted) {
        setState(() {
          isLoaded = true;
        });
      }
    } catch (e) {
      debugPrint("Error initializing camera: $e");
      // Show error to user
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Camera initialization error: $e"),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> loadYoloModel() async {
    try {
      await vision.loadYoloModel(
        labels: 'assets/AI_models/labels.txt',
        modelPath: 'assets/AI_models/cornTypeFinal104.tflite',
        modelVersion: "yolov8",
        numThreads: 1,
        useGpu: false,
      );
    } catch (e) {
      debugPrint("Error loading YOLO model: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Error loading AI model: $e"),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> startDetection() async {
    if (isDetecting || controller == null || !controller!.value.isInitialized) {
      return;
    }
    setState(() => isDetecting = true);

    await controller!.startImageStream((image) {
      if (!isDetecting) return;

      // Skip this frame if still processing previous one
      if (_isProcessing) return;

      _isProcessing = true;
      cameraImage = image;

      yoloOnFrame(image).then((_) {
        _isProcessing = false;
      }).catchError((error) {
        debugPrint("Error in YOLO processing: $error");
        _isProcessing = false;
      });
    });

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text("Detection Started!"),
            duration: Duration(seconds: 3)),
      );
    }
  }

  Future<void> stopDetection() async {
    if (controller == null) return;

    if (mounted) {
      setState(() {
        isDetecting = false;
        _isProcessing = false;
        yoloResults.clear();
      });
    }

    try {
      await controller!.stopImageStream();
    } catch (e) {
      debugPrint("Error stopping image stream: $e");
    }
  }

  void handleButtonPress() async {
    if (isDetecting) {
      await stopDetection();
    } else {
      await startDetection();
    }
  }

  Future<void> yoloOnFrame(CameraImage image) async {
    final result = await vision.yoloOnFrame(
      bytesList: image.planes.map((plane) => plane.bytes).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      iouThreshold: 0.5,
      confThreshold: 0.55,
      classThreshold: 0.55,
    );

    if (mounted && (result.isNotEmpty || yoloResults.isNotEmpty)) {
      setState(() {
        yoloResults = result;
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    stopDetection();
    controller?.dispose();
    vision.closeYoloModel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final Size screenSize = MediaQuery.of(context).size;

    if (!isLoaded || controller == null || !controller!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(controller!),
          ...displayBoxesAroundRecognizedObjects(screenSize),
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                height: 60,
                width: 60,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 2),
                  color: isDetecting ? Colors.red : Colors.green,
                ),
                child: IconButton(
                  onPressed: handleButtonPress,
                  icon: Icon(isDetecting ? Icons.stop : Icons.play_arrow),
                  color: Colors.white,
                  iconSize: 30,
                ),
              ),
            ),
          ),
          // Add status indicator
          Positioned(
            top: 40,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                isDetecting ? "Processing: ON" : "Processing: OFF",
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> displayBoxesAroundRecognizedObjects(Size screen) {
    if (yoloResults.isEmpty || cameraImage == null) return [];

    double factorX = screen.width / cameraImage!.height;
    double factorY = screen.height / cameraImage!.width;

    Map<String, Color> labelColors = {
      'white lagkitan': Colors.white,
      'sweet corn': Colors.amber,
    };

    return yoloResults.map((result) {
      double objectX = result["box"][0] * factorX;
      double objectY = result["box"][1] * factorY;
      double objectWidth = (result["box"][2] - result["box"][0]) * factorX;
      double objectHeight = (result["box"][3] - result["box"][1]) * factorY;

      String label = result['tag'];
      Color boxColor = labelColors[label.toLowerCase()] ?? Colors.grey;

      return Stack(
        children: [
          Positioned(
            left: objectX,
            top: objectY - 20,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 4.0, vertical: 2.0),
              color: boxColor.withOpacity(0.7),
              child: Text(
                "$label ${(result['box'][4] * 100).toStringAsFixed(1)}%",
                style: const TextStyle(color: Colors.black, fontSize: 10.0),
              ),
            ),
          ),
          Positioned(
            left: objectX,
            top: objectY,
            width: objectWidth,
            height: objectHeight,
            child: Container(
              decoration: BoxDecoration(
                borderRadius: const BorderRadius.all(Radius.circular(2.0)),
                border: Border.all(color: boxColor, width: 1.0),
              ),
            ),
          ),
        ],
      );
    }).toList();
  }
}
