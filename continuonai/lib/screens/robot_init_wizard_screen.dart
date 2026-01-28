import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../models/robot_init_state.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';
import '../widgets/init/wizard_step_indicator.dart';
import 'dashboard_screen.dart';

/// Multi-step wizard for initializing a new robot
class RobotInitWizardScreen extends StatefulWidget {
  static const routeName = '/robot_init_wizard';

  final BrainClient brainClient;

  const RobotInitWizardScreen({
    super.key,
    required this.brainClient,
  });

  @override
  State<RobotInitWizardScreen> createState() => _RobotInitWizardScreenState();
}

class _RobotInitWizardScreenState extends State<RobotInitWizardScreen> {
  late RobotInitState _state;
  final _nameController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  @override
  void initState() {
    super.initState();
    _state = const RobotInitState();
    _checkNetworkAndStatus();
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _checkNetworkAndStatus() async {
    setState(() {
      _state = _state.copyWith(isLoading: true, clearError: true);
    });

    try {
      // Check network connectivity
      final isLan = await widget.brainClient.checkLocalNetwork();

      // Fetch ownership status
      final ownershipData = await widget.brainClient.fetchOwnershipStatus(
        host: widget.brainClient.rcan.currentHost ?? 'localhost',
        httpPort: 8080,
      );

      final ownershipStatus = RobotOwnershipStatus.fromJson(ownershipData);

      // Fetch robot info
      final robotData = await widget.brainClient.getRobotName();
      final robotInfo = RobotInfo.fromJson(robotData);

      setState(() {
        _state = _state.copyWith(
          isLoading: false,
          networkConnected: isLan || widget.brainClient.isConnected,
          ownershipStatus: ownershipStatus,
          robotInfo: robotInfo,
          robotName: robotInfo.name,
        );
        _nameController.text = robotInfo.name;
      });
    } catch (e) {
      setState(() {
        _state = _state.copyWith(
          isLoading: false,
          error: 'Failed to connect: ${e.toString()}',
        );
      });
    }
  }

  Future<void> _claimOwnership() async {
    setState(() {
      _state = _state.copyWith(isLoading: true, clearError: true);
    });

    try {
      final user = FirebaseAuth.instance.currentUser;
      final success = await widget.brainClient.claimRobot(
        host: widget.brainClient.rcan.currentHost ?? 'localhost',
        httpPort: 8080,
        ownerId: user?.uid,
        accountId: user?.email,
        accountType: 'firebase',
      );

      if (success) {
        setState(() {
          _state = _state.copyWith(
            isLoading: false,
            claimSuccess: true,
            ownershipStatus: _state.ownershipStatus.copyWith(owned: true),
          );
        });
      } else {
        setState(() {
          _state = _state.copyWith(
            isLoading: false,
            error: 'Failed to claim robot. It may already be owned.',
          );
        });
      }
    } catch (e) {
      setState(() {
        _state = _state.copyWith(
          isLoading: false,
          error: e.toString(),
        );
      });
    }
  }

  Future<void> _saveRobotName() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _state = _state.copyWith(isLoading: true, clearError: true);
    });

    try {
      final result = await widget.brainClient.setRobotName(_nameController.text);

      if (result['success'] == true) {
        setState(() {
          _state = _state.copyWith(
            isLoading: false,
            robotName: _nameController.text,
          );
        });
        _nextStep();
      } else {
        setState(() {
          _state = _state.copyWith(
            isLoading: false,
            error: result['error'] as String? ?? 'Failed to save name',
          );
        });
      }
    } catch (e) {
      setState(() {
        _state = _state.copyWith(
          isLoading: false,
          error: e.toString(),
        );
      });
    }
  }

  Future<void> _installSeedModel() async {
    setState(() {
      _state = _state.copyWith(isLoading: true, clearError: true);
    });

    try {
      final success = await widget.brainClient.installSeedBundle(
        host: widget.brainClient.rcan.currentHost ?? 'localhost',
        httpPort: 8080,
      );

      if (success) {
        setState(() {
          _state = _state.copyWith(
            isLoading: false,
            seedInstallSuccess: true,
          );
        });
      } else {
        setState(() {
          _state = _state.copyWith(
            isLoading: false,
            error: 'Failed to install seed model',
          );
        });
      }
    } catch (e) {
      setState(() {
        _state = _state.copyWith(
          isLoading: false,
          error: e.toString(),
        );
      });
    }
  }

  void _nextStep() {
    final currentIndex = _state.currentStep.index;
    if (currentIndex < RobotInitStep.values.length - 1) {
      setState(() {
        _state = _state.copyWith(
          currentStep: RobotInitStep.fromIndex(currentIndex + 1),
          clearError: true,
        );
      });

      // Skip steps that are already completed
      if (_state.shouldSkipCurrentStep) {
        _nextStep();
      }
    }
  }

  void _previousStep() {
    final currentIndex = _state.currentStep.index;
    if (currentIndex > 0) {
      setState(() {
        _state = _state.copyWith(
          currentStep: RobotInitStep.fromIndex(currentIndex - 1),
          clearError: true,
        );
      });
    }
  }

  void _completeWizard() {
    Navigator.of(context).pushReplacementNamed(
      DashboardScreen.routeName,
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Setup Robot'),
        leading: _state.currentStep.index > 0
            ? IconButton(
                icon: const Icon(Icons.arrow_back),
                onPressed: _previousStep,
              )
            : IconButton(
                icon: const Icon(Icons.close),
                onPressed: () => Navigator.of(context).pop(),
              ),
      ),
      body: Column(
        children: [
          // Step indicator
          WizardStepIndicator(currentStep: _state.currentStep),
          const Divider(height: 1),

          // Content
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(ContinuonTokens.s24),
              child: _buildStepContent(isDark),
            ),
          ),

          // Error message
          if (_state.error != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(ContinuonTokens.s16),
              color: theme.colorScheme.errorContainer,
              child: Row(
                children: [
                  Icon(
                    Icons.error_outline,
                    color: theme.colorScheme.error,
                  ),
                  const SizedBox(width: ContinuonTokens.s8),
                  Expanded(
                    child: Text(
                      _state.error!,
                      style: TextStyle(color: theme.colorScheme.error),
                    ),
                  ),
                ],
              ),
            ),

          // Navigation buttons
          _buildNavigationButtons(isDark),
        ],
      ),
    );
  }

  Widget _buildStepContent(bool isDark) {
    switch (_state.currentStep) {
      case RobotInitStep.welcome:
        return _buildWelcomeStep(isDark);
      case RobotInitStep.claimOwnership:
        return _buildClaimStep(isDark);
      case RobotInitStep.nameRobot:
        return _buildNameStep(isDark);
      case RobotInitStep.installSeed:
        return _buildSeedStep(isDark);
      case RobotInitStep.complete:
        return _buildCompleteStep(isDark);
    }
  }

  Widget _buildWelcomeStep(bool isDark) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const SizedBox(height: ContinuonTokens.s32),
        Icon(
          Icons.smart_toy_outlined,
          size: 80,
          color: ContinuonColors.primaryBlue,
        ),
        const SizedBox(height: ContinuonTokens.s24),
        Text(
          'Welcome to Robot Setup',
          style: Theme.of(context).textTheme.headlineMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s16),
        Text(
          'Let\'s get your robot ready. This wizard will guide you through claiming ownership, naming your robot, and installing the AI runtime.',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
              ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s32),

        // Network status
        _buildStatusCard(
          icon: _state.networkConnected ? Icons.wifi : Icons.wifi_off,
          label: 'Network Connection',
          value: _state.networkConnected ? 'Connected' : 'Not Connected',
          isGood: _state.networkConnected,
          isDark: isDark,
        ),
        const SizedBox(height: ContinuonTokens.s12),

        // Robot info
        if (_state.robotInfo != null) ...[
          _buildStatusCard(
            icon: Icons.smart_toy,
            label: 'Robot Found',
            value: _state.robotInfo!.name,
            isGood: true,
            isDark: isDark,
          ),
        ],
      ],
    );
  }

  Widget _buildClaimStep(bool isDark) {
    final alreadyOwned = _state.ownershipStatus.owned;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const SizedBox(height: ContinuonTokens.s32),
        Icon(
          alreadyOwned ? Icons.verified_user : Icons.person_add,
          size: 80,
          color: alreadyOwned ? Colors.green : ContinuonColors.primaryBlue,
        ),
        const SizedBox(height: ContinuonTokens.s24),
        Text(
          alreadyOwned ? 'Robot Already Owned' : 'Claim Ownership',
          style: Theme.of(context).textTheme.headlineMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s16),
        Text(
          alreadyOwned
              ? 'This robot is already registered to your account. You can proceed to the next step.'
              : 'Register yourself as the owner of this robot. This gives you full control over updates, settings, and data.',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
              ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s32),

        if (!alreadyOwned && !_state.claimSuccess)
          ElevatedButton.icon(
            onPressed: _state.isLoading ? null : _claimOwnership,
            icon: _state.isLoading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.how_to_reg),
            label: Text(_state.isLoading ? 'Claiming...' : 'Claim This Robot'),
          ),

        if (_state.claimSuccess || alreadyOwned)
          _buildStatusCard(
            icon: Icons.check_circle,
            label: 'Ownership',
            value: 'Claimed successfully',
            isGood: true,
            isDark: isDark,
          ),
      ],
    );
  }

  Widget _buildNameStep(bool isDark) {
    return Form(
      key: _formKey,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const SizedBox(height: ContinuonTokens.s32),
          Icon(
            Icons.edit,
            size: 80,
            color: ContinuonColors.primaryBlue,
          ),
          const SizedBox(height: ContinuonTokens.s24),
          Text(
            'Name Your Robot',
            style: Theme.of(context).textTheme.headlineMedium,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: ContinuonTokens.s16),
          Text(
            'Give your robot a memorable name. This will help you identify it in your fleet.',
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                  color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
                ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: ContinuonTokens.s32),

          TextFormField(
            controller: _nameController,
            decoration: const InputDecoration(
              labelText: 'Robot Name',
              hintText: 'e.g., Kitchen Helper, Workshop Bot',
              prefixIcon: Icon(Icons.smart_toy),
              border: OutlineInputBorder(),
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter a name for your robot';
              }
              if (value.length < 2) {
                return 'Name must be at least 2 characters';
              }
              return null;
            },
            textCapitalization: TextCapitalization.words,
            onFieldSubmitted: (_) => _saveRobotName(),
          ),
        ],
      ),
    );
  }

  Widget _buildSeedStep(bool isDark) {
    final alreadyInstalled = _state.ownershipStatus.seedInstalled;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const SizedBox(height: ContinuonTokens.s32),
        Icon(
          alreadyInstalled ? Icons.psychology : Icons.cloud_download,
          size: 80,
          color: alreadyInstalled ? Colors.green : ContinuonColors.cmsViolet,
        ),
        const SizedBox(height: ContinuonTokens.s24),
        Text(
          alreadyInstalled ? 'AI Runtime Installed' : 'Install AI Runtime',
          style: Theme.of(context).textTheme.headlineMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s16),
        Text(
          alreadyInstalled
              ? 'The AI runtime is already installed on this robot. You\'re ready to go!'
              : 'Install the seed AI model that gives your robot its initial intelligence. This enables basic perception and control.',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
              ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s32),

        if (!alreadyInstalled && !_state.seedInstallSuccess)
          ElevatedButton.icon(
            onPressed: _state.isLoading ? null : _installSeedModel,
            icon: _state.isLoading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.download),
            label: Text(_state.isLoading ? 'Installing...' : 'Install Seed Model'),
          ),

        if (_state.seedInstallSuccess || alreadyInstalled)
          _buildStatusCard(
            icon: Icons.check_circle,
            label: 'AI Runtime',
            value: 'Installed successfully',
            isGood: true,
            isDark: isDark,
          ),
      ],
    );
  }

  Widget _buildCompleteStep(bool isDark) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const SizedBox(height: ContinuonTokens.s32),
        Container(
          width: 100,
          height: 100,
          decoration: BoxDecoration(
            color: Colors.green.withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: const Icon(
            Icons.celebration,
            size: 60,
            color: Colors.green,
          ),
        ),
        const SizedBox(height: ContinuonTokens.s24),
        Text(
          'Setup Complete!',
          style: Theme.of(context).textTheme.headlineMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s16),
        Text(
          'Your robot "${_state.robotName}" is ready to use. You can now control it, train it, and teach it new skills.',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
              ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: ContinuonTokens.s32),

        // Summary
        _buildStatusCard(
          icon: Icons.verified_user,
          label: 'Ownership',
          value: 'Claimed',
          isGood: true,
          isDark: isDark,
        ),
        const SizedBox(height: ContinuonTokens.s12),
        _buildStatusCard(
          icon: Icons.smart_toy,
          label: 'Robot Name',
          value: _state.robotName,
          isGood: true,
          isDark: isDark,
        ),
        const SizedBox(height: ContinuonTokens.s12),
        _buildStatusCard(
          icon: Icons.psychology,
          label: 'AI Runtime',
          value: 'Installed',
          isGood: true,
          isDark: isDark,
        ),
      ],
    );
  }

  Widget _buildStatusCard({
    required IconData icon,
    required String label,
    required String value,
    required bool isGood,
    required bool isDark,
  }) {
    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark
            ? ContinuonColors.gray800.withOpacity(0.5)
            : ContinuonColors.gray200.withOpacity(0.5),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: isGood
              ? Colors.green.withOpacity(0.3)
              : ContinuonColors.gray500.withOpacity(0.3),
        ),
      ),
      child: Row(
        children: [
          Icon(
            icon,
            color: isGood ? Colors.green : ContinuonColors.gray500,
          ),
          const SizedBox(width: ContinuonTokens.s12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 12,
                    color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
                  ),
                ),
                Text(
                  value,
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: isDark ? ContinuonColors.gray200 : ContinuonColors.gray900,
                  ),
                ),
              ],
            ),
          ),
          if (isGood)
            Icon(
              Icons.check_circle,
              color: Colors.green,
              size: 20,
            ),
        ],
      ),
    );
  }

  Widget _buildNavigationButtons(bool isDark) {
    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        border: Border(
          top: BorderSide(
            color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
          ),
        ),
      ),
      child: SafeArea(
        child: Row(
          children: [
            if (_state.currentStep.index > 0 &&
                _state.currentStep != RobotInitStep.complete)
              TextButton(
                onPressed: _previousStep,
                child: const Text('Back'),
              ),
            const Spacer(),
            if (_state.currentStep == RobotInitStep.complete)
              ElevatedButton(
                onPressed: _completeWizard,
                child: const Text('Go to Dashboard'),
              )
            else if (_state.currentStep == RobotInitStep.nameRobot)
              ElevatedButton(
                onPressed: _state.isLoading ? null : _saveRobotName,
                child: Text(_state.isLoading ? 'Saving...' : 'Save & Continue'),
              )
            else
              ElevatedButton(
                onPressed: _state.canProceed ? _nextStep : null,
                child: const Text('Continue'),
              ),
          ],
        ),
      ),
    );
  }
}

// Extension to add copyWith to RobotOwnershipStatus
extension RobotOwnershipStatusCopyWith on RobotOwnershipStatus {
  RobotOwnershipStatus copyWith({
    bool? owned,
    bool? subscriptionActive,
    bool? seedInstalled,
    String? ownerId,
    String? accountId,
    String? accountType,
  }) {
    return RobotOwnershipStatus(
      owned: owned ?? this.owned,
      subscriptionActive: subscriptionActive ?? this.subscriptionActive,
      seedInstalled: seedInstalled ?? this.seedInstalled,
      ownerId: ownerId ?? this.ownerId,
      accountId: accountId ?? this.accountId,
      accountType: accountType ?? this.accountType,
    );
  }
}
