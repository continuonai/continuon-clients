import '../models/rlds_models.dart';

class EpisodePackage {
  const EpisodePackage({required this.record, required this.assets});

  final EpisodeRecord record;
  final List<EpisodeAsset> assets;
}

class TaskRecorder {
  TaskRecorder({required this.metadata});

  EpisodeMetadata metadata;
  final List<EpisodeStep> _steps = [];
  final List<EpisodeAsset> _assets = [];

  void addStep(EpisodeStep step) {
    _steps.add(step);
  }

  void addAssets(List<EpisodeAsset> assets) {
    _assets.addAll(assets);
  }

  EpisodeRecord buildRecord({
    EpisodeMetadata? metadataOverride,
    List<EpisodeAsset>? assetsOverride,
  }) =>
      EpisodeRecord(
        metadata: metadataOverride ?? metadata,
        steps: List.of(_steps),
        assets: assetsOverride ?? List.of(_assets),
      );

  void updateMetadata(EpisodeMetadata value) {
    metadata = value;
  }

  EpisodePackage buildPackage() => EpisodePackage(
        record: buildRecord(),
        assets: List.of(_assets),
      );

  List<EpisodeAsset> markAssetsTransferred(
      Map<String, String> remoteByLocalPath) {
    final updated = _assets
        .map(
          (asset) => remoteByLocalPath.containsKey(asset.localUri)
              ? asset.withRemote(uri: remoteByLocalPath[asset.localUri]!)
              : asset,
        )
        .toList();
    _assets
      ..clear()
      ..addAll(updated);
    return updated;
  }

  void reset() {
    _steps.clear();
    _assets.clear();
  }
}
