import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:http/http.dart' as http;
import 'dart:convert' as convert;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

enum Mode { local, api, hybrid }

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _selectedImage;
  final ImagePicker _picker = ImagePicker();
  Interpreter? _interpreter;
  List<String> labels = [];

  bool _isLoadingModel = false;
  bool _isRunning = false;
  String? _errorMessage;
  List<MapEntry<String, double>> _topK = const [];
  static const int _topKCount = 3;
  Mode _mode = Mode.local;
  double _hybridThreshold = 0.50; // 0..1
  String _lastSourceLabel = '';

  String get _apiBaseUrl {
    if (Platform.isAndroid) return 'http://10.0.2.2:8000';
    return 'http://localhost:8000';
  }

  Future<List<MapEntry<String, double>>> _computeLocalTopK(File file) async {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw Exception('Interpreter not ready');
    }
    final bytes = await file.readAsBytes();
    final img.Image? originalImage = img.decodeImage(bytes);
    if (originalImage == null) {
      throw Exception('Cannot decode image');
    }
    final img.Image resizedImage = img.copyResize(originalImage, width: 224, height: 224);
    final w = resizedImage.width;
    final h = resizedImage.height;
    final Uint8List inputBytes = Uint8List(w * h * 3);
    int offset = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final pixel = resizedImage.getPixel(x, y);
        inputBytes[offset++] = pixel.r.toInt();
        inputBytes[offset++] = pixel.g.toInt();
        inputBytes[offset++] = pixel.b.toInt();
      }
    }
    var output = List.filled(1001, 0).reshape([1, 1001]);
    interpreter.run(inputBytes.reshape([1, 224, 224, 3]), output);
    final List<int> raw = List<int>.from(output[0]);
    double? scale;
    int? zeroPoint;
    try {
      final outTensor = interpreter.getOutputTensors().first;
      final params = outTensor.params;
      scale = (params.scale as num?)?.toDouble();
      zeroPoint = (params.zeroPoint as num?)?.toInt();
    } catch (_) {}
    double deq(int v) {
      if (scale != null && zeroPoint != null) {
        return ((v - zeroPoint!) * scale!).toDouble();
      }
      return v / 255.0;
    }
    final scores = <int, double>{};
    for (int i = 0; i < raw.length; i++) {
      scores[i] = deq(raw[i]);
    }
    final sorted = scores.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    final top = sorted.take(_topKCount).map((e) {
      final label = (labels.isNotEmpty && e.key < labels.length) ? labels[e.key] : 'Class ${e.key}';
      return MapEntry(label, e.value.clamp(0.0, 1.0));
    }).toList();
    return top;
  }

  Future<void> _classifyHybrid() async {
    final file = _selectedImage;
    if (file == null) return;
    setState(() {
      _isRunning = true;
      _errorMessage = null;
      _lastSourceLabel = '';
    });
    try {
      if (_interpreter == null || _isLoadingModel) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Model not ready, using Cloud')),
          );
        }
        await _classifySelectedImageCloud();
        return;
      }

      final localTop = await _computeLocalTopK(file);
      final top1 = localTop.isNotEmpty ? localTop.first.value : 0.0;

      if (top1 >= _hybridThreshold) {
        setState(() {
          _topK = localTop;
          _lastSourceLabel = 'Local (Hybrid)';
        });
      } else {
        setState(() {
          _topK = localTop;
          _lastSourceLabel = 'Local→Cloud (Hybrid)';
        });
        await _classifySelectedImageCloud();
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Hybrid failed: $e';
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_errorMessage!)),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isRunning = false;
        });
      }
    }
  }

  Uri get _predictUri => Uri.parse('$_apiBaseUrl/predict');

  @override
  void initState() {
    super.initState();
    _initLocalClassifier();
  }

  Future<void> _initLocalClassifier() async {
    setState(() {
      _isLoadingModel = true;
      _errorMessage = null;
    });
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/mobilenet_v1_1.0_224_quant.tflite',
      );
      final data = await rootBundle.loadString('assets/models/labels.txt');
      labels = data.split('\n').where((e) => e.trim().isNotEmpty).toList();
      debugPrint("Model and labels loaded successfully");
    } catch (e) {
      _errorMessage = 'Failed to initialize local model: $e';
      debugPrint(_errorMessage);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(_errorMessage!)),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoadingModel = false;
        });
      }
    }
  }

  Future<void> _pickImage() async {
    if (_mode == Mode.local && _interpreter == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model not ready yet')),
      );
      return;
    }
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        setState(() {
          _selectedImage = File(image.path);
          _topK = const [];
          _errorMessage = null;
        });
        if (_mode == Mode.local) {
          await _classifySelectedImage();
        } else if (_mode == Mode.api) {
          await _classifySelectedImageCloud();
        } else {
          await _classifyHybrid();
        }
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to pick or classify image: $e';
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_errorMessage!)),
      );
    }
  }

  Future<void> _classifySelectedImageCloud() async {
    final file = _selectedImage;
    if (file == null) return;

    setState(() {
      _isRunning = true;
      _errorMessage = null;
      _topK = const [];
    });

    try {
      // Convert to JPEG to avoid HEIC issues on backend
      final bytes = await file.readAsBytes();
      final img.Image? originalImage = img.decodeImage(bytes);
      if (originalImage == null) {
        throw Exception('Cannot decode image');
      }
      final img.Image resized = img.copyResize(originalImage, width: 224, height: 224);
      final jpg = img.encodeJpg(resized, quality: 90);

      final request = http.MultipartRequest('POST', _predictUri)
        ..files.add(
          http.MultipartFile.fromBytes(
            'file',
            jpg,
            filename: 'image.jpg',
          ),
        );

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);
      if (response.statusCode != 200) {
        throw Exception('API error ${response.statusCode}: ${response.body}');
      }
      final Map<String, dynamic> data = convert.jsonDecode(response.body) as Map<String, dynamic>;

      final List<MapEntry<String, double>> results = [];
      final predictions = data['predictions'];
      if (predictions is List) {
        for (final item in predictions) {
          final m = item as Map<String, dynamic>;
          final label = (m['label'] ?? m['class_id']).toString();
          final conf = (m['confidence'] as num?)?.toDouble() ?? 0.0;
          results.add(MapEntry(label, conf.clamp(0.0, 1.0)));
        }
      } else if (data['top_k'] is List) {
        final List list = data['top_k'];
        for (final item in list) {
          final m = item as Map<String, dynamic>;
          final label = (m['label'] ?? m['class_id']).toString();
          final conf = (m['confidence'] as num?)?.toDouble() ?? 0.0;
          results.add(MapEntry(label, conf.clamp(0.0, 1.0)));
        }
      } else {
        final label = (data['label'] ?? data['class_id']).toString();
        final conf = (data['confidence'] as num?)?.toDouble() ?? 0.0;
        results.add(MapEntry(label, conf.clamp(0.0, 1.0)));
      }

      setState(() {
        // Use the backend-provided list as-is (already top-k)
        _topK = results.toList();
        _lastSourceLabel = 'Cloud';
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Cloud inference failed: $e';
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_errorMessage!)),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isRunning = false;
        });
      }
    }
  }

  Future<void> _classifySelectedImage() async {
    final file = _selectedImage;
    final interpreter = _interpreter;
    if (file == null || interpreter == null) return;

    setState(() {
      _isRunning = true;
      _errorMessage = null;
    });

    try {
      final bytes = await file.readAsBytes();
      final img.Image? originalImage = img.decodeImage(bytes);
      if (originalImage == null) {
        throw Exception('Cannot decode image');
      }

      final img.Image resizedImage = img.copyResize(originalImage, width: 224, height: 224);
      final w = resizedImage.width;
      final h = resizedImage.height;

      final Uint8List inputBytes = Uint8List(w * h * 3);
      int offset = 0;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final pixel = resizedImage.getPixel(x, y);
          inputBytes[offset++] = pixel.r.toInt();
          inputBytes[offset++] = pixel.g.toInt();
          inputBytes[offset++] = pixel.b.toInt();
        }
      }

      var output = List.filled(1001, 0).reshape([1, 1001]);
      interpreter.run(inputBytes.reshape([1, 224, 224, 3]), output);

      final List<int> raw = List<int>.from(output[0]);

      // Try to dequantize using tensor params if available; fallback to /255
      double? scale;
      int? zeroPoint;
      try {
        final outTensor = interpreter.getOutputTensors().first;
        // Accessor names can vary; wrap in try-catch for safety
        final params = outTensor.params; // may throw
        // ignore: unnecessary_cast
        scale = (params.scale as num?)?.toDouble();
        zeroPoint = (params.zeroPoint as num?)?.toInt();
      } catch (_) {
        // Fallback handled below
      }

      double deq(int v) {
        if (scale != null && zeroPoint != null) {
          return ((v - zeroPoint!) * scale!).toDouble();
        }
        return v / 255.0; // common for uint8 quantized outputs
      }

      final scores = <int, double>{};
      for (int i = 0; i < raw.length; i++) {
        scores[i] = deq(raw[i]);
      }

      final sorted = scores.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      final top = sorted.take(_topKCount).map((e) {
        final label = (labels.isNotEmpty && e.key < labels.length)
            ? labels[e.key]
            : 'Class ${e.key}';
        return MapEntry(label, e.value.clamp(0.0, 1.0));
      }).toList();

      setState(() {
        _topK = top;
        _lastSourceLabel = 'Local';
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Inference failed: $e';
        _topK = const [];
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_errorMessage!)),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isRunning = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Image Classifier"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Mode'),
                DropdownButton<Mode>(
                  value: _mode,
                  items: const [
                    DropdownMenuItem(value: Mode.local, child: Text('Local')),
                    DropdownMenuItem(value: Mode.api, child: Text('Cloud API')),
                    DropdownMenuItem(value: Mode.hybrid, child: Text('Hybrid')),
                  ],
                  onChanged: (_isRunning)
                      ? null
                      : (m) async {
                          if (m == null) return;
                          final newMode = m;
                          setState(() {
                            _mode = newMode;
                            _topK = const [];
                            _errorMessage = null;
                          });
                          if (_selectedImage != null) {
                            if (newMode == Mode.local) {
                              if (_interpreter == null || _isLoadingModel) {
                                if (mounted) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(content: Text('Model not ready yet')),
                                  );
                                }
                              } else {
                                await _classifySelectedImage();
                              }
                            } else if (newMode == Mode.api) {
                              await _classifySelectedImageCloud();
                            } else {
                              await _classifyHybrid();
                            }
                          }
                        },
                ),
              ],
            ),
            if (_mode == Mode.api || _mode == Mode.hybrid)
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: Text(
                  'API: ${_predictUri.toString()}',
                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                ),
              ),
            if (_mode == Mode.hybrid)
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text('Hybrid threshold: ${(_hybridThreshold * 100).toStringAsFixed(0)}%'),
                    Slider(
                      value: _hybridThreshold,
                      onChanged: _isRunning ? null : (v) {
                        setState(() => _hybridThreshold = v);
                      },
                      min: 0.0,
                      max: 1.0,
                      divisions: 20,
                    ),
                  ],
                ),
              ),
            if (_isLoadingModel) ...[
              const SizedBox(height: 24),
              const Center(child: CircularProgressIndicator()),
              const SizedBox(height: 8),
              const Center(child: Text('Loading model...')),
            ] else ...[
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (_selectedImage == null)
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 24),
                          child: Center(child: Text('No image selected')),
                        )
                      else ...[
                        Image.file(
                          _selectedImage!,
                          height: 300,
                          fit: BoxFit.contain,
                        ),
                        const SizedBox(height: 16),
                      ],
                      if (_isRunning) ...[
                        const Center(child: CircularProgressIndicator()),
                        const SizedBox(height: 8),
                        const Center(child: Text('Running inference...')),
                      ] else if (_topK.isNotEmpty) ...[
                        const Text(
                          'Top predictions:',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        if (_lastSourceLabel.isNotEmpty)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 4.0),
                            child: Text(
                              'Source: $_lastSourceLabel',
                              style: const TextStyle(fontSize: 12, color: Colors.grey),
                            ),
                          ),
                        const SizedBox(height: 8),
                        for (final e in _topK)
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 4.0),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Expanded(child: Text(e.key)),
                                Text('${(e.value * 100).toStringAsFixed(1)}%'),
                              ],
                            ),
                          ),
                      ] else if (_errorMessage != null) ...[
                        Text(
                          _errorMessage!,
                          style: const TextStyle(color: Colors.red),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: (_mode == Mode.local && _isLoadingModel) || _isRunning ? null : _pickImage,
        child: const Icon(Icons.image),
      ),
    );
  }

  @override
  void dispose() {
    try {
      _interpreter?.close();
    } catch (_) {}
    super.dispose();
  }
}
