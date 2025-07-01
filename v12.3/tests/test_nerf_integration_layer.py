import pytest
import numpy as np
import asyncio
from types import SimpleNamespace

from nerf_integration_layer import (
    RenderQuality, ViewpointType, CameraParameters, RenderRequest, RenderResult, SceneMemory,
    NeRFRenderer, MockNeRFRenderer, FastRenderQueue, VisualMemoryManager, NeRFIntegrationManager
)

@pytest.mark.asyncio
async def test_mock_nerf_renderer_basic():
    renderer = MockNeRFRenderer()
    cam_params = CameraParameters(
        position=(1,2,3), orientation=(0,0,0),
        field_of_view=90.0, near_plane=0.1, far_plane=100.0, resolution=(32, 32)
    )
    req = RenderRequest(
        request_id="req1", camera_params=cam_params,
        quality=RenderQuality.LOW, viewpoint_type=ViewpointType.FIRST_PERSON,
        scene_id="sceneA"
    )
    # Test rendering
    result = await renderer.render_view(req)
    assert result.success
    assert isinstance(result.image_data, np.ndarray)
    assert result.image_data.shape == (32, 32, 3)
    # Test update_scene and get_scene_info
    assert await renderer.update_scene("sceneA", {})
    info = renderer.get_scene_info("sceneA")
    assert info["exists"]
    # Test training
    assert await renderer.train_scene("sceneA", {})

def test_camera_parameters_and_renderrequest_serialization():
    cam = CameraParameters(
        position=(0,0,0), orientation=(0,0,0), field_of_view=60.0,
        near_plane=0.5, far_plane=50.0, resolution=(100, 200)
    )
    req = RenderRequest(
        request_id="abc", camera_params=cam, quality=RenderQuality.HIGH,
        viewpoint_type=ViewpointType.OVERHEAD, scene_id="s1"
    )
    assert req.camera_params.resolution == (100, 200)
    assert req.quality == RenderQuality.HIGH
    assert req.viewpoint_type == ViewpointType.OVERHEAD

def test_scene_memory_and_features():
    cam = CameraParameters(
        position=(0,0,0), orientation=(0,0,0), field_of_view=90,
        near_plane=0.1, far_plane=100, resolution=(16,16)
    )
    dummy_img = np.zeros((16,16,3), dtype=np.uint8)
    dummy_depth = np.ones((16,16), dtype=np.float32)
    result = RenderResult(
        request_id="req", success=True, image_data=dummy_img, depth_data=dummy_depth
    )
    mem = SceneMemory(
        scene_id="x", viewpoints=[cam], rendered_views={"key": result},
        spatial_features={"key": np.ones(6)}, object_locations={"a": (1,2,3)}
    )
    assert mem.scene_id == "x"
    assert "key" in mem.rendered_views
    assert "a" in mem.object_locations

@pytest.mark.asyncio
async def test_fast_render_queue_with_mock_renderer():
    renderer = MockNeRFRenderer()
    queue = FastRenderQueue(max_queue_size=3)
    cam_params = CameraParameters(
        position=(1,1,1), orientation=(0,0,0), field_of_view=45.0,
        near_plane=0.1, far_plane=10.0, resolution=(8, 8)
    )
    req = RenderRequest(
        request_id="queue1", camera_params=cam_params,
        quality=RenderQuality.MEDIUM, viewpoint_type=ViewpointType.FIRST_PERSON,
        scene_id="sceneQ"
    )
    await queue.start_workers(renderer, num_workers=1)
    fut = await queue.submit_render_request(req)
    result = await fut
    assert result.success
    await queue.stop_workers()

def test_visual_memory_manager_create_and_evict():
    vmm = VisualMemoryManager(max_scenes=2)
    s1 = vmm.create_scene_memory("s1")
    assert s1.scene_id == "s1"
    s2 = vmm.create_scene_memory("s2")
    s3 = vmm.create_scene_memory("s3")  # Should evict one
    assert len(vmm.scene_memories) == 2
    stats = vmm.get_memory_stats()
    assert "total_scenes" in stats

def test_add_rendered_view_and_find_similar():
    vmm = VisualMemoryManager()
    cam = CameraParameters(
        position=(1,1,1), orientation=(0,0,0), field_of_view=90,
        near_plane=0.1, far_plane=10, resolution=(4,4)
    )
    result = RenderResult(
        request_id="req", success=True,
        image_data=np.zeros((4,4,3), dtype=np.uint8),
        depth_data=np.ones((4,4), dtype=np.float32)
    )
    vmm.add_rendered_view("myscene", cam, result)
    mem = vmm.get_scene_memory("myscene")
    assert mem is not None
    similars = vmm.find_similar_viewpoints("myscene", cam)
    assert len(similars) >= 1

@pytest.mark.asyncio
async def test_nerf_integration_manager_full_cycle():
    renderer = MockNeRFRenderer()
    manager = NeRFIntegrationManager(renderer)
    await manager.start()
    cam = CameraParameters(
        position=(0,0,0), orientation=(0,0,0), field_of_view=90,
        near_plane=0.1, far_plane=100, resolution=(16,16)
    )
    # Render view (first time, not in memory)
    result = await manager.render_agent_view("agentX", "sceneA", cam)
    assert result.success
    # Now should have cache hit for very close viewpoint
    cam2 = CameraParameters(
        position=(0.05, 0.05, 0.05), orientation=(0,0,0), field_of_view=90,
        near_plane=0.1, far_plane=100, resolution=(16,16)
    )
    result2 = await manager.render_agent_view("agentX", "sceneA", cam2, use_prediction=True)
    assert result2.success
    # Update scene with agent data
    agent_data = {"obs": "something"}
    assert await manager.update_scene_from_agent_data("sceneA", agent_data)
    # Get spatial understanding
    stats = manager.get_spatial_understanding("sceneA", (0,0,0))
    assert isinstance(stats, dict)
    # Get integration status
    status = manager.get_integration_status()
    assert "is_running" in status
    await manager.stop()
