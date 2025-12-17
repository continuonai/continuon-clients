class EpisodeMetadata {
  EpisodeMetadata({
    required this.xrMode,
    required this.controlRole,
    required this.environmentId,
    this.tags = const [],
    this.softwareVersions,
    this.source,
    this.provenance,
  });

  final String xrMode;
  final String controlRole;
  final String environmentId;
  final List<String> tags;
  final SoftwareVersions? softwareVersions;
  final String? source;
  final Map<String, dynamic>? provenance;

  Map<String, dynamic> toJson() => {
        'xr_mode': xrMode,
        'control_role': controlRole,
        'environment_id': environmentId,
        'tags': tags,
        if (softwareVersions != null) 'software': softwareVersions!.toJson(),
        if (source != null) 'source': source,
        if (provenance != null) 'provenance': provenance,
      };
}

class SoftwareVersions {
  const SoftwareVersions({
    this.xrApp,
    this.continuonBrainOs,
    this.gloveFirmware,
  });

  final String? xrApp;
  final String? continuonBrainOs;
  final String? gloveFirmware;

  Map<String, dynamic> toJson() => {
        if (xrApp != null) 'xr_app': xrApp,
        if (continuonBrainOs != null) 'continuonbrain_os': continuonBrainOs,
        if (gloveFirmware != null) 'glove_firmware': gloveFirmware,
      };
}

class EpisodeStep {
  EpisodeStep({
    required this.observation,
    required this.action,
    this.isTerminal = false,
    this.stepMetadata = const {},
  });

  final Map<String, dynamic> observation;
  final Map<String, dynamic> action;
  final bool isTerminal;
  final Map<String, String> stepMetadata;

  Map<String, dynamic> toJson() => {
        'observation': observation,
        'action': action,
        'is_terminal': isTerminal,
        'step_metadata': stepMetadata,
      };
}

class TaskClassification {
  const TaskClassification({
    required this.tasks,
    required this.motionPrimitives,
    this.confidence,
  });

  final List<String> tasks;
  final List<String> motionPrimitives;
  final Map<String, double>? confidence;

  Map<String, dynamic> toJson() => {
        'tasks': tasks,
        'motion_primitives': motionPrimitives,
        if (confidence != null) 'confidence': confidence,
      };
}

class EpisodeRecord {
  EpisodeRecord({
    required this.metadata,
    this.steps = const [],
    this.taskClassification,
    this.sourceContext,
  });

  final EpisodeMetadata metadata;
  final List<EpisodeStep> steps;
  final TaskClassification? taskClassification;
  final Map<String, dynamic>? sourceContext;

  Map<String, dynamic> toJson() => {
        'metadata': metadata.toJson(),
        'steps': steps.map((s) => s.toJson()).toList(),
        if (taskClassification != null)
          'task_classification': taskClassification!.toJson(),
        if (sourceContext != null) 'source_context': sourceContext,
      };
}
