//
//  ViewController.swift
//  CreateML_sample
//
//  Created by 森掛 on 2025/12/21.
//

import UIKit
import AVFoundation
import Vision
import CoreML

class ViewController: UIViewController {
    
    // MARK: - UI Components
    private let startCameraButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("カメラをオン", for: .normal)
        button.backgroundColor = .systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 10
        button.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    private let previewView: UIView = {
        let view = UIView()
        view.backgroundColor = .black
        view.translatesAutoresizingMaskIntoConstraints = false
        view.isHidden = true
        return view
    }()
    
    private let resultLabel: UILabel = {
        let label = UILabel()
        label.text = "判別結果: -"
        label.textAlignment = .center
        label.font = .systemFont(ofSize: 24, weight: .bold)
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.layer.cornerRadius = 10
        label.clipsToBounds = true
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    // MARK: - Camera Properties
    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    
    // MARK: - ML Model
    private var mlModel: JankenPoseClassifier?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupMLModel()
        setupCameraButton()
    }
    
    // MARK: - Setup Methods
    private func setupUI() {
        view.backgroundColor = .white
        
        // Add subviews
        view.addSubview(previewView)
        view.addSubview(startCameraButton)
        view.addSubview(resultLabel)
        
        // Layout constraints
        NSLayoutConstraint.activate([
            // Preview View (full screen)
            previewView.topAnchor.constraint(equalTo: view.topAnchor),
            previewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            previewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            previewView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Start Camera Button (center of screen)
            startCameraButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            startCameraButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            startCameraButton.widthAnchor.constraint(equalToConstant: 200),
            startCameraButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Result Label (top)
            resultLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            resultLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            resultLabel.widthAnchor.constraint(equalToConstant: 300),
            resultLabel.heightAnchor.constraint(equalToConstant: 60)
        ])
    }
    
    private func setupCameraButton() {
        startCameraButton.addTarget(self, action: #selector(startCameraButtonTapped), for: .touchUpInside)
    }
    
    private func setupMLModel() {
        do {
            let config = MLModelConfiguration()
            mlModel = try JankenPoseClassifier(configuration: config)
            
        } catch {
        
        }
    }
    
    // MARK: - Camera Setup
    @objc private func startCameraButtonTapped() {
        checkCameraPermissions()
    }
    
    private func checkCameraPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    DispatchQueue.main.async {
                        self?.startCamera()
                    }
                }
            }
        case .denied, .restricted:
            showPermissionAlert()
        @unknown default:
            break
        }
    }
    
    private func startCamera() {
        startCameraButton.isHidden = true
        previewView.isHidden = false
        
        let session = AVCaptureSession()
        session.sessionPreset = .high
        
        guard let videoCaptureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        
        do {
            let videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
            
            if session.canAddInput(videoInput) {
                session.addInput(videoInput)
            }
            
            // Setup video output
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            
            if session.canAddOutput(videoDataOutput) {
                session.addOutput(videoDataOutput)
            }
            
            // Setup preview layer
            let previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer.frame = previewView.bounds
            previewLayer.videoGravity = .resizeAspectFill
            previewView.layer.addSublayer(previewLayer)
            self.previewLayer = previewLayer
            
            captureSession = session
            
            DispatchQueue.global(qos: .userInitiated).async {
                session.startRunning()
            }
        } catch {
         
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = previewView.bounds
    }
    
    private func showPermissionAlert() {
        let alert = UIAlertController(
            title: "カメラへのアクセスが必要です",
            message: "設定からカメラへのアクセスを許可してください",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "設定を開く", style: .default) { _ in
            if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                UIApplication.shared.open(settingsURL)
            }
        })
        alert.addAction(UIAlertAction(title: "キャンセル", style: .cancel))
        present(alert, animated: true)
    }
    
    // MARK: - Vision Processing
    private func detectHandPose(in pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else {
            return
        }
        
        // 手のポーズを検出
        let handPoseRequest = VNDetectHumanHandPoseRequest()
        handPoseRequest.maximumHandCount = 1
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        
        do {
            try handler.perform([handPoseRequest])
            
            guard let observation = handPoseRequest.results?.first else {
                DispatchQueue.main.async { [weak self] in
                    self?.resultLabel.text = "判別結果: 手が検出されません"
                }
                return
            }
            
            // 手のキーポイントを取得してモデルに入力
            classifyHandPose(observation: observation, model: model)
        } catch {
        }
    }
    
    private func classifyHandPose(observation: VNHumanHandPoseObservation, model: JankenPoseClassifier) {
        do {
            // VNHumanHandPoseObservationをMultiArrayに変換
            let keypointsMultiArray = try observation.toMLMultiArray()
            
            // モデルで推論
            let prediction = try model.prediction(poses: keypointsMultiArray)
            
            let label = prediction.label
            let confidence = Int((prediction.labelProbabilities[label] ?? 0.0) * 100)
            let resultText = "\(label) (\(confidence)%)"
            DispatchQueue.main.async { [weak self] in
                self?.resultLabel.text = "判別結果: \(resultText)"
            }
            
        } catch {
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        detectHandPose(in: pixelBuffer)
    }
}

// MARK: - VNHumanHandPoseObservation Extension
extension VNHumanHandPoseObservation {
    func toMLMultiArray() throws -> MLMultiArray {
        // 21個のキーポイント x 3次元（x, y, confidence）= 63要素
        let multiArray = try MLMultiArray(shape: [1, 3, 21], dataType: .double)
        
        // 全てのキーポイントを取得
        let allPoints = try recognizedPoints(.all)
        
        // キーポイントの順序
        let jointNames: [VNHumanHandPoseObservation.JointName] = [
            .wrist,
            .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
            .indexMCP, .indexPIP, .indexDIP, .indexTip,
            .middleMCP, .middlePIP, .middleDIP, .middleTip,
            .ringMCP, .ringPIP, .ringDIP, .ringTip,
            .littleMCP, .littlePIP, .littleDIP, .littleTip
        ]
        
        for (index, jointName) in jointNames.enumerated() {
            guard let point = allPoints[jointName] else {
                // キーポイントが見つからない場合は0で埋める
                multiArray[[0, 0, index] as [NSNumber]] = 0.0
                multiArray[[0, 1, index] as [NSNumber]] = 0.0
                multiArray[[0, 2, index] as [NSNumber]] = 0.0
                continue
            }
            
            // x, y, confidence を設定
            multiArray[[0, 0, index] as [NSNumber]] = NSNumber(value: point.location.x)
            multiArray[[0, 1, index] as [NSNumber]] = NSNumber(value: point.location.y)
            multiArray[[0, 2, index] as [NSNumber]] = NSNumber(value: point.confidence)
        }
        
        return multiArray
    }
}

