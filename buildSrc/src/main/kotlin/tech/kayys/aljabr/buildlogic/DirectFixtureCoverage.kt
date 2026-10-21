package tech.kayys.aljabr.buildlogic

data class DirectFixtureCoverage(
    val id: String,
    val modelType: String,
    val architectures: List<String>,
    val tokenizerMarker: String,
)
