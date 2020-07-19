class Scene:

    def __init__(self, scene_size, name):
        self.scene_size = scene_size
        self.name = name
        self.buildings = []

    def add_building(self, building):
        self.buildings.append(building)

    def has_overlap(self, building):
        for structure in self.buildings:
            intersection = building.intersection(structure)
            if not intersection.is_empty:
                return True
        return False
