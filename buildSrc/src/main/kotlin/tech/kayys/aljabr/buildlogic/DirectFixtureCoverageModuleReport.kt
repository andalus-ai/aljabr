package tech.kayys.aljabr.buildlogic

data class DirectFixtureCoverageModuleReport(
    val moduleName: String,
    val fixtures: List<DirectFixtureCoverage>,
)
