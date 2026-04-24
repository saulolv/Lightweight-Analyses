package com.pibic.lightweightcnnanalyses

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                EvaluationScreen()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun EvaluationScreen() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val evaluator = remember { ModelEvaluator(context) }
    val scope = rememberCoroutineScope()

    val availableModelFiles = remember { mutableStateListOf<String>() }
    var selectedDatasetId by remember { mutableStateOf(ModelEvaluator.SUPPORTED_DATASETS.first().id) }
    var useHardwareAcceleration by remember { mutableStateOf(false) }
    val imageCountOptions = remember { listOf(100, 250, 400, 1000, 2000) }
    var maxImagesPerModel by remember { mutableStateOf(400) }

    var isRunning by remember { mutableStateOf(false) }
    var runningDataset by remember { mutableStateOf("") }
    var runningModel by remember { mutableStateOf("") }
    var processed by remember { mutableStateOf(0) }
    var total by remember { mutableStateOf(0) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var jsonFilePath by remember { mutableStateOf<String?>(null) }
    var exported by remember { mutableStateOf<Boolean?>(null) }
    var metrics by remember { mutableStateOf<List<ModelMetrics>>(emptyList()) }
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    LaunchedEffect(Unit) {
        val models = withContext(Dispatchers.IO) { evaluator.listAssetTfliteModels() }
        availableModelFiles.clear()
        availableModelFiles.addAll(models)
    }

    val selectedDataset = ModelEvaluator.SUPPORTED_DATASETS.first { it.id == selectedDatasetId }
    val pairedModels = evaluator.buildPairedModelSpecs(selectedDatasetId, availableModelFiles.toList())

    Scaffold(
        topBar = { TopAppBar(title = { Text("Benchmark TFLite") }) }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item {
                ElevatedCard {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text("Configuração", style = MaterialTheme.typography.titleMedium)
                        Text(
                            "Selecione um dataset. O app monta automaticamente os modelos pareados para a rodada.",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            ModelEvaluator.SUPPORTED_DATASETS.forEach { dataset ->
                                FilterChip(
                                    selected = selectedDatasetId == dataset.id,
                                    onClick = { selectedDatasetId = dataset.id },
                                    label = { Text(dataset.displayName) }
                                )
                            }
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text("Aceleração de hardware")
                                Text(
                                    if (useHardwareAcceleration) "Ligada (NNAPI com fallback CPU)" else "Desligada (CPU only)",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                            Switch(
                                checked = useHardwareAcceleration,
                                onCheckedChange = { useHardwareAcceleration = it }
                            )
                        }
                        Text("Amostras por modelo", style = MaterialTheme.typography.titleSmall)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            imageCountOptions.forEach { option ->
                                FilterChip(
                                    selected = maxImagesPerModel == option,
                                    onClick = { maxImagesPerModel = option },
                                    label = { Text(option.toString()) }
                                )
                            }
                        }
                        Text(
                            "Warm-up fixo: 10 inferências (não entram no cálculo de latência).",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }

            item {
                ElevatedCard {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text("Modelos pareados: ${selectedDataset.displayName}", style = MaterialTheme.typography.titleMedium)
                        if (availableModelFiles.isEmpty()) {
                            Text("Nenhum modelo .tflite encontrado em assets.", color = MaterialTheme.colorScheme.error)
                        } else if (pairedModels.isEmpty()) {
                            Text("Nenhum modelo pareado encontrado para este dataset.", color = MaterialTheme.colorScheme.error)
                        } else {
                            pairedModels.forEach { spec ->
                                ListItem(
                                    headlineContent = { Text(spec.name) },
                                    supportingContent = { Text(spec.assetFile) }
                                )
                            }
                        }
                    }
                }
            }

            item {
                ElevatedCard {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Button(
                            enabled = !isRunning && pairedModels.isNotEmpty(),
                            onClick = {
                                isRunning = true
                                errorMessage = null
                                jsonFilePath = null
                                exported = null
                                metrics = emptyList()
                                processed = 0
                                total = 0

                                scope.launch(Dispatchers.IO) {
                                    try {
                                        val result = evaluator.evaluateDataset(
                                            dataset = selectedDataset,
                                            specs = pairedModels,
                                            options = EvaluationOptions(useHardwareAcceleration = useHardwareAcceleration),
                                            maxImagesPerModel = maxImagesPerModel,
                                            warmupRuns = 10
                                        ) { datasetId, modelName, _, _, p, t ->
                                            mainHandler.post {
                                                runningDataset = datasetId
                                                runningModel = modelName
                                                processed = p
                                                total = t
                                            }
                                        }
                                        val json = evaluator.saveMetricsAsJson(result)
                                        val exportOk = evaluator.exportToDownloads(json)
                                        mainHandler.post {
                                            metrics = result
                                            jsonFilePath = json.absolutePath
                                            exported = exportOk
                                        }
                                    } catch (e: Exception) {
                                        mainHandler.post {
                                            errorMessage = e.message ?: "Falha na avaliação."
                                        }
                                    } finally {
                                        mainHandler.post { isRunning = false }
                                    }
                                }
                            }
                        ) {
                            Icon(Icons.Filled.PlayArrow, contentDescription = null)
                            Text("Iniciar avaliação")
                        }

                        if (isRunning) {
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator()
                                Column {
                                    Text("Executando: $runningDataset")
                                    Text("Modelo: $runningModel")
                                }
                            }
                            LinearProgressIndicator(
                                progress = { if (total > 0) processed.toFloat() / total.toFloat() else 0f },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Text("$processed / $total imagens")
                        }

                        if (jsonFilePath != null) {
                            Button(
                                enabled = !isRunning,
                                onClick = {
                                    scope.launch(Dispatchers.IO) {
                                        val path = jsonFilePath
                                        if (path != null) {
                                            val result = evaluator.exportToDownloads(File(path))
                                            mainHandler.post { exported = result }
                                        }
                                    }
                                }
                            ) {
                                Text("Exportar JSON novamente")
                            }
                        }

                        errorMessage?.let { Text("Erro: $it", color = MaterialTheme.colorScheme.error) }
                        exported?.let { ok ->
                            Text(
                                if (ok) "JSON exportado para Downloads" else "Falha ao exportar JSON",
                                color = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
                            )
                        }
                        jsonFilePath?.let { Text("Arquivo interno: $it") }
                    }
                }
            }

            item {
                Text("Resultados", style = MaterialTheme.typography.titleMedium)
            }
            if (metrics.isEmpty()) {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 24.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("Nenhum resultado disponível.")
                    }
                }
            } else {
                items(metrics) { m ->
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = if (m.evalStatus == "ok") {
                                MaterialTheme.colorScheme.surfaceVariant
                            } else {
                                MaterialTheme.colorScheme.errorContainer
                            }
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            Text("${m.datasetName} - ${m.modelName}", style = MaterialTheme.typography.titleSmall)
                            Text("Status: ${m.evalStatus}")
                            m.errorMessage?.let { Text("Detalhe: $it", color = MaterialTheme.colorScheme.error) }
                            Text("Arquivo: ${m.modelFile}")
                            Text("Delegate: ${m.effectiveDelegate} | HW: ${m.useHardwareAcceleration}")
                            Text("Imagens: ${m.imagesProcessed} (falhas: ${m.imagesFailedToLoad})")
                            Text("Acurácia: ${m.accuracy?.let { String.format("%.4f", it) } ?: "N/A"}")
                            Text("Warm-up usado: ${m.warmupRunsUsed}")
                            Text(
                                "Latência (ms): média ${String.format("%.2f", m.avgLatencyMs)}, " +
                                    "min ${String.format("%.2f", m.minLatencyMs)}, " +
                                    "mediana ${String.format("%.2f", m.medianLatencyMs)}, " +
                                    "p90 ${String.format("%.2f", m.p90LatencyMs)}, " +
                                    "max ${String.format("%.2f", m.maxLatencyMs)}"
                            )
                            Text("Desvio padrão latência: ${String.format("%.2f", m.latencyStdMs)} ms")
                            Text("Tempo total inferência: ${String.format("%.2f", m.totalTimeMs)} ms")
                            Text("Throughput: ${String.format("%.2f", m.throughputImagesPerSecond)} img/s")
                        }
                    }
                }
            }
        }
    }
}
