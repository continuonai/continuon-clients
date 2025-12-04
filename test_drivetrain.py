import sys
import time
from continuonbrain.actuators.drivetrain_controller import DrivetrainController

def test_drivetrain():
    print("Testing DrivetrainController...")
    
    try:
        controller = DrivetrainController()
        success = controller.initialize()
        
        if not success:
            print("❌ Failed to initialize controller")
            return
            
        print(f"✅ Controller initialized (Mode: {controller.mode})")
        
        if controller.mode == "mock":
            print("⚠️  Running in MOCK mode (hardware not detected/library missing)")
            return

        print("Testing steering...")
        controller.apply_drive(steering=0.5, throttle=0.0)
        time.sleep(1)
        controller.apply_drive(steering=-0.5, throttle=0.0)
        time.sleep(1)
        controller.apply_drive(steering=0.0, throttle=0.0)
        
        print("Testing throttle...")
        controller.apply_drive(steering=0.0, throttle=0.3)
        time.sleep(1)
        controller.apply_drive(steering=0.0, throttle=0.0)
        
        print("✅ Test complete")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_drivetrain()
