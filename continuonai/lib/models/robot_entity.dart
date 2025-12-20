import 'package:equatable/equatable.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// Represents a Robot Entity in Firestore (collection: 'robots').
/// Mirrors the structure of the local `settings.json` and identity state.
class RobotEntity extends Equatable {
  final String id;
  final String ownerId;
  final String name;
  final bool isPublic;
  final bool online;
  final DateTime? lastSeen;
  final String mode;
  final Map<String, dynamic> settings;
  final List<String> capabilities;
  final Map<String, dynamic> hardware;

  const RobotEntity({
    required this.id,
    required this.ownerId,
    required this.name,
    this.isPublic = false,
    this.online = false,
    this.lastSeen,
    this.mode = 'idle',
    this.settings = const {},
    this.capabilities = const [],
    this.hardware = const {},
  });

  /// Creates a [RobotEntity] from a Firestore document snapshot.
  factory RobotEntity.fromSnapshot(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>? ?? {};
    return RobotEntity(
      id: doc.id,
      ownerId: data['ownerId'] as String? ?? '',
      name: data['name'] as String? ?? 'Unnamed Robot',
      isPublic: data['isPublic'] as bool? ?? false,
      online: data['online'] as bool? ?? false,
      lastSeen: (data['lastSeen'] as Timestamp?)?.toDate(),
      mode: data['mode'] as String? ?? 'idle',
      settings: data['settings'] as Map<String, dynamic>? ?? {},
      capabilities: List<String>.from(data['capabilities'] ?? []),
      hardware: data['hardware'] as Map<String, dynamic>? ?? {},
    );
  }

  /// Converts the [RobotEntity] to a Map for Firestore storage.
  Map<String, dynamic> toDocument() {
    return {
      'ownerId': ownerId,
      'name': name,
      'isPublic': isPublic,
      'online': online,
      'lastSeen': lastSeen != null ? Timestamp.fromDate(lastSeen!) : null,
      'mode': mode,
      'settings': settings,
      'capabilities': capabilities,
      'hardware': hardware,
    };
  }

  /// Creates a copy of this RobotEntity with the given fields replaced with the new values.
  RobotEntity copyWith({
    String? id,
    String? ownerId,
    String? name,
    bool? isPublic,
    bool? online,
    DateTime? lastSeen,
    String? mode,
    Map<String, dynamic>? settings,
    List<String>? capabilities,
    Map<String, dynamic>? hardware,
  }) {
    return RobotEntity(
      id: id ?? this.id,
      ownerId: ownerId ?? this.ownerId,
      name: name ?? this.name,
      isPublic: isPublic ?? this.isPublic,
      online: online ?? this.online,
      lastSeen: lastSeen ?? this.lastSeen,
      mode: mode ?? this.mode,
      settings: settings ?? this.settings,
      capabilities: capabilities ?? this.capabilities,
      hardware: hardware ?? this.hardware,
    );
  }

  @override
  List<Object?> get props => [
        id,
        ownerId,
        name,
        isPublic,
        online,
        lastSeen,
        mode,
        settings,
        capabilities,
        hardware,
      ];
}
