import 'dart:convert' as convert;
import 'dart:typed_data';
import 'package:dio/dio.dart';

class HfApiResult {
  final List<MapEntry<String, double>> predictions;
  final int downBytes;
  const HfApiResult({required this.predictions, required this.downBytes});
}

class HfApi {
  static const String endpoint =
      'https://router.huggingface.co/hf-inference/models/google/vit-base-patch16-224';

  static final Dio _dio = Dio(
    BaseOptions(
      responseType: ResponseType.bytes,
      connectTimeout: const Duration(seconds: 20),
      receiveTimeout: const Duration(seconds: 30),
      sendTimeout: const Duration(seconds: 30),
      headers: {
        'Accept': 'application/json',
      },
    ),
  );

  static Future<HfApiResult> classify(
    Uint8List imageBytes, {
    required String token,
    String? endpoint,
  }) async {
    if (token.isEmpty) {
      throw Exception('HF token is missing');
    }

    final uri = Uri.parse(endpoint ?? HfApi.endpoint);
    final resp = await _dio.post<List<int>>(
      uri.toString(),
      data: imageBytes,
      options: Options(
        contentType: 'application/octet-stream',
        headers: {
          'Authorization': 'Bearer $token',
        },
      ),
    );

    final bytes = resp.data ?? const <int>[];
    final down = bytes.length;
    final status = resp.statusCode ?? 0;
    if (status != 200) {
      final text = bytes.isEmpty ? '' : convert.utf8.decode(bytes);
      throw Exception('HF error $status: $text');
    }

    final decodedText = convert.utf8.decode(bytes);
    final decoded = convert.jsonDecode(decodedText);
    if (decoded is! List) {
      throw Exception('Unexpected HF response: $decodedText');
    }
    final preds = <MapEntry<String, double>>[];
    for (final item in decoded) {
      final m = item as Map<String, dynamic>;
      final label = (m['label'] ?? '').toString();
      final score = (m['score'] as num?)?.toDouble() ?? 0.0;
      preds.add(MapEntry(label, score.clamp(0.0, 1.0)));
    }
    return HfApiResult(predictions: preds, downBytes: down);
  }
}
