package com.anonymous.lightweightcnnanalyses

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.BatteryManager
import android.os.Process
import android.provider.MediaStore
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt
import kotlin.system.measureNanoTime

data class EvaluationOptions(
    val useHardwareAcceleration: Boolean = false,
    val numThreads: Int = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
)

enum class Normalization {
    AS_IS_FLOAT32,
    DIV_255,
    ZERO_CENTER_UNIT_RANGE,
    MOBILENET_V3
}

data class DatasetSpec(
    val id: String,
    val displayName: String,
    val rootAssetDir: String,
    val csvFileName: String,
    val imageDirName: String = "images_png"
)

data class ModelSpec(
    val name: String,
    val assetFile: String,
    val normalization: Normalization = Normalization.AS_IS_FLOAT32
)

data class EvaluationSample(
    val assetPath: String,
    val label: Int,
    val className: String? = null
)

data class ModelMetrics(
    val datasetId: String,
    val datasetName: String,
    val modelName: String,
    val modelFile: String,
    val modelSizeBytes: Long,
    val csvFile: String,
    val imagesProcessed: Int,
    val imagesFailedToLoad: Int,
    val labelsMatchedForAccuracy: Int,
    val accuracy: Double?,
    val avgLatencyMs: Double,
    val minLatencyMs: Double,
    val maxLatencyMs: Double,
    val medianLatencyMs: Double,
    val p90LatencyMs: Double,
    val latencyStdMs: Double,
    val throughputImagesPerSecond: Double,
    val totalTimeMs: Double,
    val warmupRunsUsed: Int,
    val inputShape: IntArray,
    val outputShape: IntArray,
    val memoryUsageBeforeBytes: Long,
    val memoryUsageAfterBytes: Long,
    val batteryEnergyBeforeUWh: Long?,
    val batteryEnergyAfterUWh: Long?,
    val batteryEnergyDeltaUWh: Long?,
    val processCpuTimeDeltaMs: Long,
    val useHardwareAcceleration: Boolean,
    val effectiveDelegate: String,
    val fallbackReason: String?,
    val evalStatus: String,
    val errorMessage: String?
)

class ModelEvaluator(private val context: Context) {
    companion object {
        val SUPPORTED_DATASETS = listOf(
            DatasetSpec(
                id = "cifar10",
                displayName = "CIFAR-10",
                rootAssetDir = "export_cifar10",
                csvFileName = "labels.csv"
            ),
            DatasetSpec(
                id = "cifar100",
                displayName = "CIFAR-100",
                rootAssetDir = "export_cifar100",
                csvFileName = "labels.csv"
            ),
            DatasetSpec(
                id = "wakevision",
                displayName = "WakeVision",
                rootAssetDir = "export_wakevision",
                csvFileName = "labels.csv"
            )
        )

        private val PAIRED_MODEL_KEYWORDS = mapOf(
            "cifar10" to listOf("cifar10"),
            "cifar100" to listOf("cifar100"),
            "wakevision" to listOf("wakevision", "wake_vision", "wake")
        )
    }

    private data class EnergySnapshot(val energyUWh: Long?)

    private fun usedMemory(): Long {
        val rt = Runtime.getRuntime()
        return rt.totalMemory() - rt.freeMemory()
    }

    private fun getBatteryEnergyUWh(): Long? = try {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager ?: return null
        val value = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER)
        if (value == Long.MIN_VALUE) null else value
    } catch (_: Exception) {
        null
    }

    private fun captureEnergySnapshot() = EnergySnapshot(energyUWh = getBatteryEnergyUWh())

    fun buildPairedModelSpecs(datasetId: String, modelFiles: List<String>): List<ModelSpec> {
        val keywords = PAIRED_MODEL_KEYWORDS[datasetId].orEmpty()
        if (keywords.isEmpty()) return emptyList()

        fun matchesKeyword(fileName: String, keyword: String): Boolean {
            val name = fileName.lowercase()
            val kw = keyword.lowercase()
            // Evita falso positivo como "cifar10" em "cifar100" usando bordas por caracteres não alfanuméricos.
            val pattern = Regex("(^|[^a-z0-9])${Regex.escape(kw)}([^a-z0-9]|$)")
            return pattern.containsMatchIn(name)
        }

        return modelFiles
            .filter { file -> keywords.any { kw -> matchesKeyword(file, kw) } }
            .sorted()
            .map { file ->
                ModelSpec(
                    name = File(file).nameWithoutExtension,
                    assetFile = file,
                    normalization = Normalization.AS_IS_FLOAT32
                )
            }
    }

    fun listAssetTfliteModels(): List<String> {
        return try {
            context.assets.list("")?.filter { it.endsWith(".tflite", ignoreCase = true) }?.sorted()
                ?: emptyList()
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun listDatasetSamples(dataset: DatasetSpec): List<EvaluationSample> {
        val csvPath = "${dataset.rootAssetDir}/${dataset.csvFileName}"
        return try {
            context.assets.open(csvPath).bufferedReader().use { reader ->
                val rows = reader.readLines()
                if (rows.isEmpty()) return emptyList()

                val header = rows.first().split(",").map { it.trim() }
                val filenameIdx = header.indexOfFirst { it.equals("filename", ignoreCase = true) }
                val labelIdx = header.indexOfFirst { it.equals("label", ignoreCase = true) }
                val classNameIdx = header.indexOfFirst {
                    it.equals("class_name", ignoreCase = true) || it.equals("label_name", ignoreCase = true)
                }

                if (filenameIdx < 0 || labelIdx < 0) {
                    Log.e("ModelEvaluator", "CSV inválido: falta coluna filename e/ou label em $csvPath")
                    return emptyList()
                }

                rows.drop(1).mapNotNull { line ->
                    val cols = line.split(",")
                    if (cols.size <= maxOf(filenameIdx, labelIdx)) return@mapNotNull null
                    val filename = cols[filenameIdx].trim()
                    val label = cols[labelIdx].trim().toIntOrNull() ?: return@mapNotNull null
                    val className = if (classNameIdx >= 0 && cols.size > classNameIdx) {
                        cols[classNameIdx].trim()
                    } else {
                        null
                    }
                    EvaluationSample(
                        assetPath = "${dataset.rootAssetDir}/${dataset.imageDirName}/$filename",
                        label = label,
                        className = className
                    )
                }
            }
        } catch (e: Exception) {
            Log.e("ModelEvaluator", "Falha ao carregar CSV ${dataset.displayName}: ${e.message}")
            emptyList()
        }
    }

    private fun loadBitmap(assetPath: String): Bitmap? = try {
        context.assets.open(assetPath).use { BitmapFactory.decodeStream(it) }
    } catch (_: Exception) {
        null
    }

    private fun createInterpreter(modelBytes: ByteArray, options: EvaluationOptions): Triple<Interpreter, String, String?> {
        val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(modelBytes)
        modelBuffer.rewind()
        if (!options.useHardwareAcceleration) {
            val cpuOptions = Interpreter.Options().apply {
                setNumThreads(options.numThreads)
            }
            return Triple(Interpreter(modelBuffer, cpuOptions), "CPU", null)
        }

        // Tenta NNAPI; em caso de falha, faz fallback para CPU.
        return try {
            val nnapiOptions = Interpreter.Options().apply {
                setNumThreads(options.numThreads)
                setUseNNAPI(true)
            }
            Triple(Interpreter(modelBuffer, nnapiOptions), "NNAPI", null)
        } catch (e: Exception) {
            Log.w("ModelEvaluator", "Falha ao iniciar NNAPI, fallback CPU: ${e.message}")
            val cpuFallback = Interpreter.Options().apply {
                setNumThreads(options.numThreads)
                setUseNNAPI(false)
            }
            Triple(Interpreter(modelBuffer, cpuFallback), "CPU_FALLBACK", e.message ?: "unknown")
        }
    }

    private fun normalize(value: Float, mode: Normalization): Float = when (mode) {
        Normalization.AS_IS_FLOAT32 -> value
        Normalization.DIV_255 -> value / 255f
        Normalization.ZERO_CENTER_UNIT_RANGE -> (value / 255f - 0.5f) * 2f
        Normalization.MOBILENET_V3 -> (value - 127.5f) / 127.5f
    }

    private fun preprocess(bitmap: Bitmap, inputShape: IntArray, inputType: DataType, mode: Normalization): Any {
        val (h, w, c) = when (inputShape.size) {
            4 -> Triple(inputShape[1], inputShape[2], inputShape[3])
            3 -> Triple(inputShape[0], inputShape[1], inputShape[2])
            else -> throw IllegalArgumentException("Input shape não suportado: ${inputShape.contentToString()}")
        }
        val resized = Bitmap.createScaledBitmap(bitmap, w, h, true)
        return when (inputType) {
            DataType.FLOAT32 -> {
                val buffer = ByteBuffer.allocateDirect(4 * h * w * c).order(ByteOrder.nativeOrder())
                val pixels = IntArray(w * h)
                resized.getPixels(pixels, 0, w, 0, 0, w, h)
                var idx = 0
                repeat(h) {
                    repeat(w) {
                        val px = pixels[idx++]
                        val r = normalize((px shr 16 and 0xFF).toFloat(), mode)
                        val g = normalize((px shr 8 and 0xFF).toFloat(), mode)
                        val b = normalize((px and 0xFF).toFloat(), mode)
                        when (c) {
                            1 -> buffer.putFloat(0.299f * r + 0.587f * g + 0.114f * b)
                            3 -> {
                                buffer.putFloat(r)
                                buffer.putFloat(g)
                                buffer.putFloat(b)
                            }
                            else -> {
                                buffer.putFloat(r)
                                buffer.putFloat(g)
                                buffer.putFloat(b)
                            }
                        }
                    }
                }
                buffer.rewind()
                buffer
            }
            DataType.UINT8 -> {
                val buffer = ByteBuffer.allocateDirect(h * w * c).order(ByteOrder.nativeOrder())
                val pixels = IntArray(w * h)
                resized.getPixels(pixels, 0, w, 0, 0, w, h)
                var idx = 0
                repeat(h) {
                    repeat(w) {
                        val px = pixels[idx++]
                        val r = (px shr 16 and 0xFF).toByte()
                        val g = (px shr 8 and 0xFF).toByte()
                        val b = (px and 0xFF).toByte()
                        when (c) {
                            1 -> {
                                val gray = (((r.toInt() and 0xFF) * 0.299) + ((g.toInt() and 0xFF) * 0.587) + ((b.toInt() and 0xFF) * 0.114))
                                    .toInt().coerceIn(0, 255)
                                buffer.put(gray.toByte())
                            }
                            3 -> {
                                buffer.put(r)
                                buffer.put(g)
                                buffer.put(b)
                            }
                            else -> {
                                buffer.put(r)
                                buffer.put(g)
                                buffer.put(b)
                            }
                        }
                    }
                }
                buffer.rewind()
                buffer
            }
            else -> throw IllegalArgumentException("Tipo de input não suportado: $inputType")
        }
    }

    private fun outputContainer(outputShape: IntArray, outputType: DataType): Any {
        val outputSize = outputShape.reduce { acc, v -> acc * v }
        return when (outputType) {
            DataType.FLOAT32 -> if (outputShape.size == 2 && outputShape[0] == 1) {
                Array(1) { FloatArray(outputShape[1]) }
            } else {
                FloatArray(outputSize)
            }
            DataType.UINT8 -> if (outputShape.size == 2 && outputShape[0] == 1) {
                Array(1) { ByteArray(outputShape[1]) }
            } else {
                ByteArray(outputSize)
            }
            else -> throw IllegalArgumentException("Tipo de output não suportado: $outputType")
        }
    }

    private fun getScores(output: Any, outputType: DataType): FloatArray? {
        return when (outputType) {
            DataType.FLOAT32 -> when (output) {
                is FloatArray -> output
                is Array<*> -> output.firstOrNull() as? FloatArray
                else -> null
            }
            DataType.UINT8 -> {
                val bytes: ByteArray? = when (output) {
                    is ByteArray -> output
                    is Array<*> -> output.firstOrNull() as? ByteArray
                    else -> null
                }
                bytes?.map { (it.toInt() and 0xFF).toFloat() }?.toFloatArray()
            }
            else -> null
        }
    }

    private fun buildErrorMetrics(
        dataset: DatasetSpec,
        spec: ModelSpec,
        modelSizeBytes: Long,
        outputShape: IntArray = intArrayOf(),
        inputShape: IntArray = intArrayOf(),
        options: EvaluationOptions,
        effectiveDelegate: String,
        fallbackReason: String? = null,
        status: String,
        errorMessage: String?
    ): ModelMetrics {
        return ModelMetrics(
            datasetId = dataset.id,
            datasetName = dataset.displayName,
            modelName = spec.name,
            modelFile = spec.assetFile,
            modelSizeBytes = modelSizeBytes,
            csvFile = "${dataset.rootAssetDir}/${dataset.csvFileName}",
            imagesProcessed = 0,
            imagesFailedToLoad = 0,
            labelsMatchedForAccuracy = 0,
            accuracy = null,
            avgLatencyMs = 0.0,
            minLatencyMs = 0.0,
            maxLatencyMs = 0.0,
            medianLatencyMs = 0.0,
            p90LatencyMs = 0.0,
            latencyStdMs = 0.0,
            throughputImagesPerSecond = 0.0,
            totalTimeMs = 0.0,
            warmupRunsUsed = 0,
            inputShape = inputShape,
            outputShape = outputShape,
            memoryUsageBeforeBytes = 0L,
            memoryUsageAfterBytes = 0L,
            batteryEnergyBeforeUWh = null,
            batteryEnergyAfterUWh = null,
            batteryEnergyDeltaUWh = null,
            processCpuTimeDeltaMs = 0L,
            useHardwareAcceleration = options.useHardwareAcceleration,
            effectiveDelegate = effectiveDelegate,
            fallbackReason = fallbackReason,
            evalStatus = status,
            errorMessage = errorMessage
        )
    }

    fun evaluateDataset(
        dataset: DatasetSpec,
        specs: List<ModelSpec>,
        options: EvaluationOptions = EvaluationOptions(),
        maxImagesPerModel: Int? = null,
        warmupRuns: Int = 5,
        progressCallback: (datasetId: String, modelName: String, modelIdx: Int, modelTotal: Int, processed: Int, total: Int) -> Unit
    ): List<ModelMetrics> {
        val samplesRaw = listDatasetSamples(dataset)
        val samples = if (maxImagesPerModel != null) samplesRaw.take(maxImagesPerModel) else samplesRaw
        if (samples.isEmpty()) return emptyList()
        val maxLabel = samples.maxOfOrNull { it.label } ?: -1

        val metrics = mutableListOf<ModelMetrics>()
        val modelsTotal = specs.size

        specs.forEachIndexed { index, spec ->
            var processed = 0
            var failedLoads = 0
            var matchedForAccuracy = 0
            var correct = 0

            val modelBytes = try {
                context.assets.open(spec.assetFile).use { it.readBytes() }
            } catch (e: Exception) {
                Log.e("ModelEvaluator", "Falha ao ler modelo ${spec.assetFile}: ${e.message}")
                metrics += buildErrorMetrics(
                    dataset = dataset,
                    spec = spec,
                    modelSizeBytes = 0L,
                    options = options,
                    effectiveDelegate = "N/A",
                    fallbackReason = null,
                    status = "model_load_error",
                    errorMessage = e.message
                )
                return@forEachIndexed
            }

            val interpreterAndDelegate = try {
                createInterpreter(modelBytes, options)
            } catch (e: Exception) {
                Log.e("ModelEvaluator", "Falha ao criar interpreter ${spec.assetFile}: ${e.message}")
                metrics += buildErrorMetrics(
                    dataset = dataset,
                    spec = spec,
                    modelSizeBytes = modelBytes.size.toLong(),
                    options = options,
                    effectiveDelegate = "N/A",
                    fallbackReason = null,
                    status = "interpreter_init_error",
                    errorMessage = e.message
                )
                return@forEachIndexed
            }
            val interpreter = interpreterAndDelegate.first
            val effectiveDelegate = interpreterAndDelegate.second
            val fallbackReason = interpreterAndDelegate.third

            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            val inputShape = inputTensor.shape()
            val outputShape = outputTensor.shape()
            val inputType = inputTensor.dataType()
            val outputType = outputTensor.dataType()
            val output = outputContainer(outputShape, outputType)
            val outputClasses = outputShape.lastOrNull() ?: 0

            if (outputClasses <= maxLabel) {
                interpreter.close()
                metrics += buildErrorMetrics(
                    dataset = dataset,
                    spec = spec,
                    modelSizeBytes = modelBytes.size.toLong(),
                    outputShape = outputShape,
                    inputShape = inputShape,
                    options = options,
                    effectiveDelegate = effectiveDelegate,
                    fallbackReason = fallbackReason,
                    status = "incompatible_output_classes",
                    errorMessage = "Modelo tem $outputClasses classes, dataset requer rótulo máximo $maxLabel."
                )
                return@forEachIndexed
            }

            if (warmupRuns > 0) {
                val warmBitmap = loadBitmap(samples.first().assetPath)
                if (warmBitmap != null) {
                    val warmInput = preprocess(warmBitmap, inputShape, inputType, spec.normalization)
                    repeat(warmupRuns) { interpreter.run(warmInput, output) }
                    warmBitmap.recycle()
                }
            }

            val memBefore = usedMemory()
            val energyBefore = captureEnergySnapshot()
            val cpuBefore = Process.getElapsedCpuTime()

            val latencies = mutableListOf<Double>()
            val totalNs = measureNanoTime {
                samples.forEach { sample ->
                    val bmp = loadBitmap(sample.assetPath)
                    if (bmp == null) {
                        failedLoads++
                        processed++
                        progressCallback(dataset.id, spec.name, index + 1, modelsTotal, processed, samples.size)
                        return@forEach
                    }

                    val input = preprocess(bmp, inputShape, inputType, spec.normalization)
                    val latencyNs = measureNanoTime {
                        interpreter.run(input, output)
                    }
                    bmp.recycle()

                    latencies += latencyNs / 1_000_000.0
                    val scores = getScores(output, outputType)
                    if (scores != null && outputClasses > 0 && sample.label in 0 until outputClasses) {
                        val pred = scores.indices.maxByOrNull { scores[it] } ?: -1
                        if (pred >= 0) {
                            matchedForAccuracy++
                            if (pred == sample.label) correct++
                        }
                    }
                    processed++
                    progressCallback(dataset.id, spec.name, index + 1, modelsTotal, processed, samples.size)
                }
            }

            val cpuAfter = Process.getElapsedCpuTime()
            val energyAfter = captureEnergySnapshot()
            val memAfter = usedMemory()
            interpreter.close()

            val sorted = latencies.sorted()
            fun percentile(p: Double): Double {
                if (sorted.isEmpty()) return 0.0
                val idx = ((p / 100.0) * (sorted.size - 1)).coerceIn(0.0, (sorted.size - 1).toDouble())
                val lo = idx.toInt()
                val hi = minOf(lo + 1, sorted.size - 1)
                val frac = idx - lo
                return sorted[lo] * (1 - frac) + sorted[hi] * frac
            }

            val avg = if (latencies.isNotEmpty()) latencies.average() else 0.0
            val std = if (latencies.size > 1) {
                sqrt(latencies.sumOf { (it - avg) * (it - avg) } / (latencies.size - 1))
            } else {
                0.0
            }
            val accuracy = if (matchedForAccuracy > 0) correct.toDouble() / matchedForAccuracy else null

            metrics += ModelMetrics(
                datasetId = dataset.id,
                datasetName = dataset.displayName,
                modelName = spec.name,
                modelFile = spec.assetFile,
                modelSizeBytes = modelBytes.size.toLong(),
                csvFile = "${dataset.rootAssetDir}/${dataset.csvFileName}",
                imagesProcessed = processed,
                imagesFailedToLoad = failedLoads,
                labelsMatchedForAccuracy = matchedForAccuracy,
                accuracy = accuracy,
                avgLatencyMs = avg,
                minLatencyMs = latencies.minOrNull() ?: 0.0,
                maxLatencyMs = latencies.maxOrNull() ?: 0.0,
                medianLatencyMs = percentile(50.0),
                p90LatencyMs = percentile(90.0),
                latencyStdMs = std,
                throughputImagesPerSecond = if (totalNs > 0) processed / (totalNs / 1_000_000_000.0) else 0.0,
                totalTimeMs = totalNs / 1_000_000.0,
                warmupRunsUsed = warmupRuns,
                inputShape = inputShape,
                outputShape = outputShape,
                memoryUsageBeforeBytes = memBefore,
                memoryUsageAfterBytes = memAfter,
                batteryEnergyBeforeUWh = energyBefore.energyUWh,
                batteryEnergyAfterUWh = energyAfter.energyUWh,
                batteryEnergyDeltaUWh = if (energyBefore.energyUWh != null && energyAfter.energyUWh != null) {
                    energyAfter.energyUWh - energyBefore.energyUWh
                } else {
                    null
                },
                processCpuTimeDeltaMs = cpuAfter - cpuBefore,
                useHardwareAcceleration = options.useHardwareAcceleration,
                effectiveDelegate = effectiveDelegate,
                fallbackReason = fallbackReason,
                evalStatus = "ok",
                errorMessage = null
            )
        }

        return metrics
    }

    fun saveMetricsAsJson(metrics: List<ModelMetrics>): File {
        val root = JSONObject()
        val arr = JSONArray()
        metrics.forEach { m ->
            arr.put(
                JSONObject()
                    .put("datasetId", m.datasetId)
                    .put("datasetName", m.datasetName)
                    .put("modelName", m.modelName)
                    .put("modelFile", m.modelFile)
                    .put("modelSizeBytes", m.modelSizeBytes)
                    .put("csvFile", m.csvFile)
                    .put("imagesProcessed", m.imagesProcessed)
                    .put("imagesFailedToLoad", m.imagesFailedToLoad)
                    .put("labelsMatchedForAccuracy", m.labelsMatchedForAccuracy)
                    .put("accuracy", m.accuracy)
                    .put("avgLatencyMs", m.avgLatencyMs)
                    .put("minLatencyMs", m.minLatencyMs)
                    .put("maxLatencyMs", m.maxLatencyMs)
                    .put("medianLatencyMs", m.medianLatencyMs)
                    .put("p90LatencyMs", m.p90LatencyMs)
                    .put("latencyStdMs", m.latencyStdMs)
                    .put("throughputImagesPerSecond", m.throughputImagesPerSecond)
                    .put("totalTimeMs", m.totalTimeMs)
                    .put("warmupRunsUsed", m.warmupRunsUsed)
                    .put("inputShape", JSONArray(m.inputShape.toList()))
                    .put("outputShape", JSONArray(m.outputShape.toList()))
                    .put("memoryUsageBeforeBytes", m.memoryUsageBeforeBytes)
                    .put("memoryUsageAfterBytes", m.memoryUsageAfterBytes)
                    .put("batteryEnergyBeforeUWh", m.batteryEnergyBeforeUWh)
                    .put("batteryEnergyAfterUWh", m.batteryEnergyAfterUWh)
                    .put("batteryEnergyDeltaUWh", m.batteryEnergyDeltaUWh)
                    .put("processCpuTimeDeltaMs", m.processCpuTimeDeltaMs)
                    .put("useHardwareAcceleration", m.useHardwareAcceleration)
                    .put("effectiveDelegate", m.effectiveDelegate)
                    .put("fallbackReason", m.fallbackReason)
                    .put("evalStatus", m.evalStatus)
                    .put("errorMessage", m.errorMessage)
            )
        }
        root.put("timestamp", System.currentTimeMillis())
        root.put("models", arr)

        val datasetId = metrics.firstOrNull()?.datasetId ?: "unknown"
        val hwAcc = metrics.firstOrNull()?.useHardwareAcceleration == true
        val hwDelegate = metrics.firstOrNull()?.effectiveDelegate ?: "unknown"
        val hwStr = if (hwAcc) {
            if (hwDelegate.equals("CPU_FALLBACK", ignoreCase = true)) {
                 "cpu_fallback"
            } else {
                 "gpu_nnapi_${hwDelegate.lowercase()}"
            }
        } else {
            "cpu"
        }
        val maxImagesStr = metrics.firstOrNull()?.imagesProcessed?.let { "_${it}imgs" } ?: ""

        val file = File(context.filesDir, "metrics_${datasetId}_${hwStr}${maxImagesStr}_${System.currentTimeMillis()}.json")
        FileOutputStream(file).use { it.write(root.toString(2).toByteArray()) }
        return file
    }

    fun exportToDownloads(jsonFile: File): Boolean {
        return try {
            val resolver = context.contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, jsonFile.name)
                put(MediaStore.Downloads.MIME_TYPE, "application/json")
                put(MediaStore.Downloads.IS_PENDING, 1)
            }
            val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return false
            resolver.openOutputStream(uri)?.use { out ->
                jsonFile.inputStream().use { input -> input.copyTo(out) }
            }
            values.clear()
            values.put(MediaStore.Downloads.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            true
        } catch (e: Exception) {
            Log.e("ModelEvaluator", "Falha ao exportar JSON: ${e.message}")
            false
        }
    }
}
