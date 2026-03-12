import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'services/disease_api_service.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Disease Scanner',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.green),
      home: const ScanPage(),
    );
  }
}

class ScanPage extends StatefulWidget {
  const ScanPage({super.key});
  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  File? _image;
  DiseaseResult? _result;
  bool _loading = false;
  String? _error;
  bool _showExplanations = true; // Toggle LIME+SHAP on/off

  Future<void> _pickImage(ImageSource source) async {
    final picked = await ImagePicker().pickImage(source: source);
    if (picked == null) return;
    setState(() {
      _image = File(picked.path);
      _result = null;
      _error = null;
    });
    await _analyze();
  }

  Future<void> _analyze() async {
    if (_image == null) return;
    setState(() => _loading = true);
    try {
      final result = await DiseaseApiService.predictDisease(
        _image!,
        includeExplanations: _showExplanations,
      );
      setState(() => _result = result);
    } on DiseaseApiException catch (e) {
      setState(() => _error = e.message);
    } finally {
      setState(() => _loading = false);
    }
  }

  Color _severityColor(String level) {
    switch (level) {
      case 'Mild':     return Colors.green;
      case 'Moderate': return Colors.orange;
      case 'Severe':   return Colors.red;
      case 'Critical': return Colors.purple;
      default:         return Colors.grey;
    }
  }

  // ── Show base64 image (LIME or SHAP) ──────────
  Widget _explanationImage(String? b64, String title, String subtitle, Color color) {
    if (b64 == null) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 16),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: color.withOpacity(0.08),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withOpacity(0.4)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title,
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color)),
              const SizedBox(height: 2),
              Text(subtitle,
                style: const TextStyle(fontSize: 12, color: Colors.grey)),
              const SizedBox(height: 10),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.memory(
                  base64Decode(b64),
                  width: double.infinity,
                  fit: BoxFit.contain,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ── All Predictions Progress Bars ─────────────
  Widget _allPredictions(Map<String, double> preds) {
    final sorted = preds.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final maxVal = sorted.first.value;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 12),
        const Divider(),
        const Text('📈 All Disease Probabilities',
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        ...sorted.map((entry) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 3),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(entry.key,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: entry.value == maxVal
                            ? FontWeight.bold : FontWeight.normal,
                        color: entry.value == maxVal
                            ? Colors.green[800] : Colors.grey[700],
                      )),
                  ),
                  Text('${entry.value.toStringAsFixed(1)}%',
                    style: const TextStyle(fontSize: 12)),
                ],
              ),
              const SizedBox(height: 2),
              LinearProgressIndicator(
                value: entry.value / 100,
                backgroundColor: Colors.grey[200],
                valueColor: AlwaysStoppedAnimation<Color>(
                  entry.value == maxVal ? Colors.green : Colors.green[200]!,
                ),
                minHeight: 5,
                borderRadius: BorderRadius.circular(4),
              ),
            ],
          ),
        )),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🌿 Crop Disease Scanner'),
        centerTitle: true,
        actions: [
          // Toggle LIME+SHAP
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Row(
              children: [
                const Text('AI\nExplain',
                  style: TextStyle(fontSize: 10),
                  textAlign: TextAlign.center),
                Switch(
                  value: _showExplanations,
                  onChanged: (v) => setState(() => _showExplanations = v),
                  activeColor: Colors.green,
                ),
              ],
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [

            // ── Image Preview ──
            Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey[300]!),
              ),
              child: _image != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(_image!, fit: BoxFit.cover),
                    )
                  : const Center(
                      child: Text('No image selected',
                          style: TextStyle(color: Colors.grey)),
                    ),
            ),
            const SizedBox(height: 20),

            // ── Buttons ──
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _loading ? null : () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _loading ? null : () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // ── Loading ──
            if (_loading)
              Column(
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 10),
                  Text(
                    _showExplanations
                        ? 'Analyzing + generating AI explanations...\n(~15 seconds)'
                        : 'Analyzing leaf...',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.grey),
                  ),
                ],
              ),

            // ── Error ──
            if (_error != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red[50],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(_error!, style: const TextStyle(color: Colors.red)),
              ),

            // ── Results ──
            if (_result != null) ...[

              // Main result card (your original design)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.green[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.green[200]!),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Results',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const Divider(),
                    Text('🦠 Disease: ${_result!.disease}',
                      style: const TextStyle(fontSize: 15)),
                    const SizedBox(height: 8),
                    Text('📊 Confidence: ${_result!.confidence.toStringAsFixed(1)}%',
                      style: const TextStyle(fontSize: 15)),
                    const SizedBox(height: 8),
                    Text('🔬 Severity: ${_result!.severityPercentage}%',
                      style: const TextStyle(fontSize: 15)),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        const Text('⚠️ Level: ', style: TextStyle(fontSize: 15)),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 4),
                          decoration: BoxDecoration(
                            color: _severityColor(_result!.severityLevel),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(_result!.severityLevel,
                            style: const TextStyle(
                                color: Colors.white, fontWeight: FontWeight.bold)),
                        ),
                      ],
                    ),

                    // All predictions chart
                    if (_result!.allPredictions.isNotEmpty)
                      _allPredictions(_result!.allPredictions),
                  ],
                ),
              ),

              // ── LIME Explanation ──
              _explanationImage(
                _result!.limeImage,
                '🟢 LIME Explanation',
                'Shows WHICH parts of the leaf caused the detection',
                Colors.green,
              ),

              // ── SHAP Explanation ──
              _explanationImage(
                _result!.shapImage,
                '🔥 SHAP Explanation',
                'Shows HOW MUCH each pixel influenced the AI decision',
                Colors.orange,
              ),

              const SizedBox(height: 24),
            ],
          ],
        ),
      ),
    );
  }
}
