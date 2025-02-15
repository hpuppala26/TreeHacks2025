from panda3d.core import Point3
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

class FlightSimulator(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the airplane model
        self.plane = self.loader.loadModel("./airplane.egg")
        self.plane.reparentTo(self.render)
        self.plane.setScale(0.5, 0.5, 0.5)  # Scale it properly
        self.plane.setPos(0, 10, 1)  # Set initial position above ground

        # Load a simple ground (runway)
        self.ground = self.loader.loadModel("models/environment")
        self.ground.reparentTo(self.render)
        self.ground.setScale(0.25, 0.25, 0.25)
        self.ground.setPos(-8, 42, -1)

        # Setup camera
        self.disableMouse()
        self.camera.setPos(0, -20, 5)  # Behind the plane
        self.camera.lookAt(self.plane)

        # Add plane movement task
        self.taskMgr.add(self.update_movement, "update_movement")

    def update_movement(self, task):
        # Simple forward movement
        self.plane.setY(self.plane, 0.1)
        return Task.cont

if __name__ == "__main__":
    app = FlightSimulator()
    app.run()
