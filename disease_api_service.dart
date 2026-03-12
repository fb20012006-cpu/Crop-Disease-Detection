import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

// ─────────────────────────────────────────────
// Data Model — includes LIME and SHAP images
// ─────────────────────────────────────────────
class DiseaseResult {
  final String disease;
  final double confidence;
  final double severityPercentage;
  final String severityLevel;
  final Map<String, double> allPredictions;
  final String? limeImage;   // base64 PNG
  final String? shapImage;   // base64 PNG

  DiseaseResult({
    required this.disease,
    required this.confidence,
    required this.severityPercentage,
    required this.severityLevel,
    required this.allPredictions,
    this.limeImage,
    this.shapImage,
  });

  factory DiseaseResult.fromJson(Map<String, dynamic> json) {
    // Parse all_predictions map
    final rawPreds = json['all_predictions'] as Map<String, dynamic>? ?? {};
    final allPreds = rawPreds.map(
      (key, value) => MapEntry(key, (value as num).toDouble()),
    );

    return DiseaseResult(
      disease: json['disease'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      severityPercentage: (json['severity_percentage'] as num).toDouble(),
      severityLevel: json['severity_level'] as String,
      allPredictions: allPreds,
      limeImage: json['lime_image'] as String?,
      shapImage: json['shap_image'] as String?,
    );
  }
}

// ─────────────────────────────────────────────
// API Service
// ─────────────────────────────────────────────
class DiseaseApiService {
  // ⚠️ CHANGE THIS URL:
  // Local WiFi:  'http://10.225.17.175:5000'
  // Cloud:       'https://your-app-name.onrender.com'
  static const String _baseUrl = 'http://10.225.17.175:5000';
  static const Duration _timeout = Duration(seconds: 60); // longer for LIME+SHAP

  /// Predict disease from image file.
  /// Set [includeExplanations] to true to get LIME + SHAP images.
  static Future<DiseaseResult> predictDisease(
    File imageFile, {
    bool includeExplanations = true,
  }) async {
    final uri = Uri.parse(
      '$_baseUrl/predict?explain=${includeExplanations ? "true" : "false"}',
    );

    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      await http.MultipartFile.fromPath('image', imageFile.path),
    );

    try {
      final streamedResponse = await request.send().timeout(_timeout);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body) as Map<String, dynamic>;
        return DiseaseResult.fromJson(json);
      } else {
        final error = jsonDecode(response.body)['error'] ?? 'Unknown error';
        throw DiseaseApiException(
          'API Error ${response.statusCode}: $error',
          statusCode: response.statusCode,
        );
      }
    } on SocketException {
      throw DiseaseApiException('No internet connection or server unreachable.');
    } on http.ClientException catch (e) {
      throw DiseaseApiException('HTTP error: ${e.message}');
    }
  }

  static Future<bool> healthCheck() async {
    try {
      final response = await http
          .get(Uri.parse('$_baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}

// ─────────────────────────────────────────────
// Custom Exception
// ─────────────────────────────────────────────
class DiseaseApiException implements Exception {
  final String message;
  final int? statusCode;
  DiseaseApiException(this.message, {this.statusCode});

  @override
  String toString() => 'DiseaseApiException: $message';
}
