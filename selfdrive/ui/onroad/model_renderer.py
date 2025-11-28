import colorsys
import numpy as np
import pyray as rl
from cereal import messaging, car
from dataclasses import dataclass, field
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient
from openpilot.system.ui.widgets import Widget

from collections import deque


CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0

THROTTLE_COLORS = [
  rl.Color(13, 248, 122, 102),   # HSLF(148/360, 0.94, 0.51, 0.4)
  rl.Color(114, 255, 92, 89),    # HSLF(112/360, 1.0, 0.68, 0.35)
  rl.Color(114, 255, 92, 0),     # HSLF(112/360, 1.0, 0.68, 0.0)
]

NO_THROTTLE_COLORS = [
  rl.Color(242, 242, 242, 102), # HSLF(148/360, 0.0, 0.95, 0.4)
  rl.Color(242, 242, 242, 89),  # HSLF(112/360, 0.0, 0.95, 0.35)
  rl.Color(242, 242, 242, 0),   # HSLF(112/360, 0.0, 0.95, 0.0)
]

# kisa
LL_THROTTLE_COLORS = [
  rl.Color(64, 200, 255, 102),
  rl.Color(90, 220, 255, 89),
  rl.Color(90, 220, 255, 0),
]

LL_NO_THROTTLE_COLORS = [
  rl.Color(200, 230, 255, 102),
  rl.Color(200, 230, 255, 89),
  rl.Color(200, 230, 255, 0),
]

@dataclass
class ModelPoints:
  raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
  projected_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))


@dataclass
class LeadVehicle:
  glow: list[float] = field(default_factory=list)
  chevron: list[float] = field(default_factory=list)
  fill_alpha: int = 0


class DrawPlot:
  PLOT_MAX = 400

  def __init__(self):
    self.plotSize = 0
    self.plotIndex = 0
    self.plotQueue = [[0.0]*self.PLOT_MAX for _ in range(3)]
    self.plotMin = 0.0
    self.plotMax = 0.0
    self.plotX = 350.0
    self.plotY = 40.0
    self.plotHeight = 300.0
    self.plotDx = 2.0
    self.plotRatio = 1.0
    self.show_plot_mode_prev = -1
    self.minDeque = [deque() for _ in range(3)]
    self.maxDeque = [deque() for _ in range(3)]

  def _draw_plotting(self, renderer, index, start, x, y_list, size, color, stroke=2):
    span = (self.plotMax - self.plotMin)
    self.plotRatio = self.plotHeight if span < 1.0 else (self.plotHeight / span)
    dx = self.plotDx

    if size <= 0:
      return

    prev_x = None
    prev_y = None
    for i in range(size):
      data = y_list[(start - i) % self.PLOT_MAX]
      plot_y = self.plotY + self.plotHeight - (data - self.plotMin) * self.plotRatio
      x_pos = x + (size - i) * dx

      if prev_x is not None:
        # pyray wrapper uses rl.draw_line
        try:
          rl.draw_line(int(prev_x), int(prev_y), int(x_pos), int(plot_y), color)
          if stroke > 1:
            rl.draw_line(int(prev_x), int(prev_y)+1, int(x_pos), int(plot_y)+1, color)
        except Exception:
          # 일부 환경에서 함수명/시그니처 다를 수 있으니 예외 무시
          pass
      else:
        # 첫 점 값 텍스트 출력 (위치 조정)
        txt = "{:.2f}".format(data)
        y_offset = 40 if index > 0 else 0
        try:
          rl.draw_text(txt, int(x_pos + 50), int(plot_y + y_offset), 14, rl.Color(255,255,255,255))
        except Exception:
          pass

      prev_x = x_pos
      prev_y = plot_y

  def make_plot_data(self, renderer):
    sm = ui_state.sm
    try:
      car_state = sm["carState"]
      lp = sm['longitudinalPlan']
      car_control = sm['carControl']
      controls_state = sm['controlsState']
      torque_state = controls_state.lateralControlState.torqueState
      a_ego = car_state.aEgo
      v_ego = car_state.vEgo
      accel = lp.accels[0]
      speeds_0 = lp.speeds[0]
      accel_out = car_control.actuators.accel
      model = sm['modelV2'].modelV2
      position = model.position
      velocity = model.velocity
      live_params = sm['liveParameters'].liveParameters
    except Exception:
      return [0.0, 0.0, 0.0], "no data"

    m = ui_state.show_plot_mode
    data = [0.0, 0.0, 0.0]
    title = "no data"

    if m in (0, 1):
      data[0] = a_ego
      data[1] = accel
      data[2] = accel_out
      title = "1.Accel (Y:a_ego, G:a_target, O:a_out)"
    elif m == 2:
      data[0] = speeds_0
      data[1] = v_ego
      data[2] = a_ego
      title = "2.Speed/Accel(Y:speed_0, G:v_ego, O:a_ego)"
    elif m == 3:
      try:
        data[0] = position.x[32]
      except Exception:
        data[0] = 0.0
      try:
        data[1] = velocity.x[32]
        data[2] = velocity.x[0]
      except Exception:
        data[1] = data[2] = 0.0
      title = "3.Model(Y:pos_32, G:vel_32, O:vel_0)"
    elif m == 4:
      data[0] = accel
      if sm.valid['radarState']:
        lead = sm['radarState'].radarState.leadOne
        data[1] = lead.aLeadK if lead is not None else 0.0
        data[2] = lead.vRel if lead is not None else 0.0
      title = "4.Lead(Y:accel, G:a_lead, O:v_rel)"
    elif m == 5:
      data[0] = a_ego
      if sm.valid['radarState']:
        lead = sm['radarState'].radarState.leadOne
        data[1] = lead.aLead if lead else 0.0
        data[2] = lead.jLead if lead else 0.0
      title = "5.Lead(Y:a_ego, G:a_lead, O:j_lead)"
    elif m == 6:
      data[0] = torque_state.actualLateralAccel * 10.0
      data[1] = torque_state.desiredLateralAccel * 10.0
      data[2] = torque_state.output * 10.0
      title = "6.Steer(Y:actual, G:desire, O:output)"
    elif m == 7:
      data[0] = car_state.steeringAngleDeg
      data[1] = car_control.actuators.steeringAngleDeg
      data[2] = live_params.angleOffsetDeg * 10.0
      title = "7.SteerA (Y:Actual, G:Target, O:Offset*10)"
    elif m == 8:
      try:
        curv = car_control.actuators.curvature * 10000.0
      except Exception:
        curv = 0.0
      data = [curv, curv, curv]
      title = "8.SteerA (Y:Actual, G:Target, O:Offset*10)"
    else:
      data = [0.0, 0.0, 0.0]
      title = "no data"

    if ui_state.show_plot_mode != self.show_plot_mode_prev:
      self.plotSize = 0
      self.plotIndex = 0
      self.plotMin = 0.0
      self.plotMax = 0.0
      for i in range(3):
        self.minDeque[i].clear()
        self.maxDeque[i].clear()
      self.show_plot_mode_prev = ui_state.show_plot_mode

    return data, title

  def update_plot_queue(self, plot_data):
    self.plotIndex = (self.plotIndex + 1) % self.PLOT_MAX
    for i in range(3):
      if self.plotSize == self.PLOT_MAX:
        if self.minDeque[i] and self.minDeque[i][0] == self.plotQueue[i][self.plotIndex]:
          self.minDeque[i].popleft()
        if self.maxDeque[i] and self.maxDeque[i][0] == self.plotQueue[i][self.plotIndex]:
          self.maxDeque[i].popleft()

      self.plotQueue[i][self.plotIndex] = plot_data[i]

      while self.minDeque[i] and self.minDeque[i][-1] > plot_data[i]:
        self.minDeque[i].pop()
      self.minDeque[i].append(plot_data[i])

      while self.maxDeque[i] and self.maxDeque[i][-1] < plot_data[i]:
        self.maxDeque[i].pop()
      self.maxDeque[i].append(plot_data[i])

    if self.plotSize < self.PLOT_MAX:
      self.plotSize += 1

    self.plotMin = float('inf')
    self.plotMax = -float('inf')
    for i in range(3):
      if self.minDeque[i]:
        self.plotMin = min(self.plotMin, self.minDeque[i][0])
      if self.maxDeque[i]:
        self.plotMax = max(self.plotMax, self.maxDeque[i][0])

    # 최소/최대 범위 고정
    if self.plotMin > -2.0:
      self.plotMin = -2.0
    if self.plotMax < 2.0:
      self.plotMax = 2.0

  def draw(self, renderer):
    # renderer: ModelRenderer 인스턴스 (self)
    if ui_state.show_plot_mode == 0:
      return

    sm = ui_state.sm
    if not (sm.alive('carState') and sm.alive('longitudinalPlan')):
      return

    plot_data, title = self.make_plot_data(renderer)
    self.update_plot_queue(plot_data)

    if getattr(renderer, "_rect", None) is None:
      return
    if renderer._rect.width < 1200:
      return

    # 색상 (파일 상단 스타일 유지)
    COLOR_YELLOW = rl.Color(240, 200, 0, 255)
    COLOR_GREEN  = rl.Color(0, 200, 100, 255)
    COLOR_ORANGE = rl.Color(255, 128, 0, 255)
    COLOR_WHITE  = rl.Color(255,255,255,255)
    colors = [COLOR_YELLOW, COLOR_GREEN, COLOR_ORANGE]

    # 각 채널 그리기
    for i in range(3):
      self._draw_plotting(renderer, i, self.plotIndex, self.plotX, self.plotQueue[i], self.plotSize, colors[i], stroke=2)

    # 제목 출력
    try:
      rl.draw_text(title, int(self.plotX + 400), int(self.plotY - 20), 18, COLOR_WHITE)
    except Exception:
      pass


class ModelRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_filter = FirstOrderFilter(1.0, 0.25, 1 / gui_app.target_fps)
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self._path_offset_z = HEIGHT_INIT[0]

    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._transform_dirty = True
    self._clip_region = None

    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=[],
      stops=[],
    )

    # Get longitudinal control setting from car parameters
    if car_params := Params().get("CarParams"):
      cp = messaging.log_from_bytes(car_params, car.CarParams)
      self._longitudinal_control = cp.openpilotLongitudinalControl

    self._draw_plot = DrawPlot()

  def set_transform(self, transform: np.ndarray):
    self._car_space_transform = transform.astype(np.float32)
    self._transform_dirty = True

  def _render(self, rect: rl.Rectangle):
    sm = ui_state.sm

    # Check if data is up-to-date
    if (sm.recv_frame["liveCalibration"] < ui_state.started_frame or
        sm.recv_frame["modelV2"] < ui_state.started_frame):
      return

    # Set up clipping region
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # Update state
    self._experimental_mode = sm['selfdriveState'].experimentalMode

    live_calib = sm['liveCalibration']
    self._path_offset_z = live_calib.height[0] if live_calib.height else HEIGHT_INIT[0]

    if sm.updated['carParams']:
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    model = sm['modelV2']
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state else None
    render_lead_indicator = radar_state is not None

    # Update model data when needed
    model_updated = sm.updated['modelV2']
    if model_updated or sm.updated['radarState'] or self._transform_dirty:
      if model_updated:
        self._update_raw_points(model)

      path_x_array = self._path.raw_points[:, 0]
      if path_x_array.size == 0:
        return

      self._update_model(lead_one, path_x_array)
      if render_lead_indicator:
        self._update_leads(radar_state, path_x_array)
      self._transform_dirty = False

    # Draw elements
    self._draw_lane_lines()
    self._draw_path(sm)

    try:
      self._draw_plot.draw(self)
    except Exception:
      pass

    if render_lead_indicator and radar_state:
      self._draw_lead_indicator()
      self._draw_radar_info(radar_state)

  def _update_raw_points(self, model):
    """Update raw 3D points from model data"""
    self._path.raw_points = np.array([model.position.x, model.position.y, model.position.z], dtype=np.float32).T

    for i, lane_line in enumerate(model.laneLines):
      self._lane_lines[i].raw_points = np.array([lane_line.x, lane_line.y, lane_line.z], dtype=np.float32).T

    for i, road_edge in enumerate(model.roadEdges):
      self._road_edges[i].raw_points = np.array([road_edge.x, road_edge.y, road_edge.z], dtype=np.float32).T

    self._lane_line_probs = np.array(model.laneLineProbs, dtype=np.float32)
    self._road_edge_stds = np.array(model.roadEdgeStds, dtype=np.float32)
    self._acceleration_x = np.array(model.acceleration.x, dtype=np.float32)

  def _update_leads(self, radar_state, path_x_array):
    """Update positions of lead vehicles"""
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    leads = [radar_state.leadOne, radar_state.leadTwo]

    for i, lead_data in enumerate(leads):
      if lead_data and lead_data.status:
        d_rel, y_rel, v_rel = lead_data.dRel, lead_data.yRel, lead_data.vRel
        idx = self._get_path_length_idx(path_x_array, d_rel)

        # Get z-coordinate from path at the lead vehicle position
        z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
        point = self._map_to_screen(d_rel, -y_rel, z + self._path_offset_z)
        if point:
          self._lead_vehicles[i] = self._update_lead_vehicle(d_rel, v_rel, point, self._rect)

  def _update_model(self, lead, path_x_array):
    """Update model visualization data based on model message"""
    max_distance = np.clip(path_x_array[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = self._get_path_length_idx(self._lane_lines[0].raw_points[:, 0], max_distance)

    # Update lane lines using raw points
    for i, lane_line in enumerate(self._lane_lines):
      lane_line.projected_points = self._map_line_to_polygon(
        lane_line.raw_points, 0.025 * self._lane_line_probs[i], 0.0, max_idx, max_distance
      )

    # Update road edges using raw points
    for road_edge in self._road_edges:
      road_edge.projected_points = self._map_line_to_polygon(road_edge.raw_points, 0.025, 0.0, max_idx, max_distance)

    # Update path using raw points
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)

    max_idx = self._get_path_length_idx(path_x_array, max_distance)
    self._path.projected_points = self._map_line_to_polygon(
      self._path.raw_points, 0.9, self._path_offset_z, max_idx, max_distance, allow_invert=False
    )

    self._update_experimental_gradient()

  def _update_experimental_gradient(self):
    """Pre-calculate experimental mode gradient colors"""
    if not self._experimental_mode:
      return

    max_len = min(len(self._path.projected_points) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      # Some points (screen space) are out of frame (rect space)
      track_y = self._path.projected_points[i][1]
      if track_y < self._rect.y or track_y > (self._rect.y + self._rect.height):
        i += 1
        continue

      # Calculate color based on acceleration (0 is bottom, 1 is top)
      lin_grad_point = 1 - (track_y - self._rect.y) / self._rect.height

      # speed up: 120, slow down: 0
      path_hue = np.clip(60 + self._acceleration_x[i] * 35, 0, 120)

      saturation = min(abs(self._acceleration_x[i] * 1.5), 1)
      lightness = np.interp(saturation, [0.0, 1.0], [0.95, 0.62])
      alpha = np.interp(lin_grad_point, [0.75 / 2.0, 0.75], [0.4, 0.0])

      # Use HSL to RGB conversion
      color = self._hsla_to_color(path_hue / 360.0, saturation, lightness, alpha)

      gradient_stops.append(lin_grad_point)
      segment_colors.append(color)

      # Skip a point, unless next is last
      i += 1 + (1 if (i + 2) < max_len else 0)

    # Store the gradient in the path object
    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=segment_colors,
      stops=gradient_stops,
    )

  def _update_lead_vehicle(self, d_rel, v_rel, point, rect):
    speed_buff, lead_buff = 10.0, 40.0

    # Calculate fill alpha
    fill_alpha = 0
    if d_rel < lead_buff:
      fill_alpha = 255 * (1.0 - (d_rel / lead_buff))
      if v_rel < 0:
        fill_alpha += 255 * (-1 * (v_rel / speed_buff))
      fill_alpha = min(fill_alpha, 255)

    # Calculate size and position
    sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 2.35
    x = np.clip(point[0], 0.0, rect.width - sz / 2)
    y = min(point[1], rect.height - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    glow = [(x + (sz * 1.35) + g_xo, y + sz + g_yo), (x, y - g_yo), (x - (sz * 1.35) - g_xo, y + sz + g_yo)]
    chevron = [(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)]

    return LeadVehicle(glow=glow, chevron=chevron, fill_alpha=int(fill_alpha))

  def _draw_lane_lines(self):
    """Draw lane lines, road edges and bsm alert"""
    for i, lane_line in enumerate(self._lane_lines):
      if lane_line.projected_points.size == 0:
        continue

      alpha = np.clip(self._lane_line_probs[i], 0.0, 0.7)
      color = rl.Color(0, 255, 100, int(alpha * 255))
      draw_polygon(self._rect, lane_line.projected_points, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size == 0:
        continue

      alpha = np.clip(1.0 - self._road_edge_stds[i], 0.0, 1.0)
      color = rl.Color(255, 0, 0, int(alpha * 255))
      draw_polygon(self._rect, road_edge.projected_points, color)

    if ui_state.show_ui_bsm:
      if ui_state.leftblindspot:
        if len(self._lane_lines) >= 2:
          # 0 ~ 1
          left_pts = self._lane_lines[0].projected_points
          mid_pts  = self._lane_lines[1].projected_points
          if left_pts.size > 0 and mid_pts.size > 0:
            polygon_pts = np.vstack([left_pts, mid_pts])
            draw_polygon(self._rect, polygon_pts, rl.Color(230, 50, 50, 125))
          # e0 ~ 1
          elif len(self._road_edges) >= 1:
            road_edge_pts = self._road_edges[0].projected_points
            if mid_pts.size > 0 and road_edge_pts.size > 0:
              polygon_pts = np.vstack([mid_pts, road_edge_pts])
              draw_polygon(self._rect, polygon_pts, rl.Color(230, 50, 50, 125))

      if ui_state.rightblindspot:
        if len(self._lane_lines) >= 4:
          # 2 ~ 3
          right_bsm = False
          mid_pts = self._lane_lines[2].projected_points
          right_pts = self._lane_lines[3].projected_points
          if mid_pts.size > 0 and right_pts.size > 0:
            right_bsm = True
            polygon_pts = np.vstack([mid_pts, right_pts])
            draw_polygon(self._rect, polygon_pts, rl.Color(230, 50, 50, 125))
            return
        # 2 ~ e1
        if len(self._lane_lines) >= 3 and len(self._road_edges) >= 2 and not right_bsm:
          mid_pts = self._lane_lines[2].projected_points
          road_edge_pts = self._road_edges[1].projected_points
          if mid_pts.size > 0 and road_edge_pts.size > 0:
            polygon_pts = np.vstack([mid_pts, road_edge_pts])
            draw_polygon(self._rect, polygon_pts, rl.Color(230, 50, 50, 125))

  def _draw_path(self, sm):
    """Draw path with dynamic coloring based on mode and throttle state."""
    if not self._path.projected_points.size:
      return

    allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control
    self._blend_filter.update(int(allow_throttle))

    if self._experimental_mode:
      # Draw with acceleration coloring
      if len(self._exp_gradient.colors) > 1:
        draw_polygon(self._rect, self._path.projected_points, gradient=self._exp_gradient)
      else:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(255, 255, 255, 30))
    else:
      if not (ui_state.enabled or ui_state.latEnabled):
        draw_polygon(self._rect, self._path.projected_points, rl.Color(255, 255, 255, 30))
      else:
        # Blend throttle/no throttle colors based on transition
        blend_factor = round(self._blend_filter.x * 100) / 100
        if ui_state.activeLaneLine:
          blended_colors = self._blend_colors(NO_THROTTLE_COLORS, THROTTLE_COLORS, blend_factor)
        else:
          blended_colors = self._blend_colors(LL_NO_THROTTLE_COLORS, LL_THROTTLE_COLORS, blend_factor)
        gradient = Gradient(
          start=(0.0, 1.0),  # Bottom of path
          end=(0.0, 0.0),  # Top of path
          colors=blended_colors,
          stops=[0.0, 0.5, 1.0],
        )
        draw_polygon(self._rect, self._path.projected_points, gradient=gradient)

  def _draw_lead_indicator(self):
    # Draw lead vehicles if available
    for lead in self._lead_vehicles:
      if not lead.glow or not lead.chevron:
        continue

      if 0 < ui_state.radarDRel < 149:
        rl.draw_triangle_fan(lead.glow, len(lead.glow), rl.Color(218, 202, 37, 255))
        rl.draw_triangle_fan(lead.chevron, len(lead.chevron), rl.Color(201, 34, 49, lead.fill_alpha))
      else:
        rl.draw_triangle_fan(lead.glow, len(lead.glow), rl.Color(100, 255, 100, 255))
        rl.draw_triangle_fan(lead.chevron, len(lead.chevron), rl.Color(50, 200, 50, lead.fill_alpha))

  def _draw_radar_info(self, radar_state):
    """Draw radar details.

    - Uses Params ShowRadarInfo (0/1/2/3) to control verbosity
    - Uses RadarLatFactor to compute prediction time (percent /100)
    - Renders boxes with relative speed, predicted point line and circle, and optional distance/lateral text
    """
    s = ui_state

    try:
      raw_lat = s.radar_lat_factor
      radar_lat_factor = float(raw_lat) / 100.0 if raw_lat is not None else 0.2
    except Exception:
      radar_lat_factor = 0.2

    if s.show_radar_info <= 0:
      return

    # We'll render for leadOne and leadTwo if present
    leads = []
    if radar_state is None:
      return

    if getattr(radar_state, 'leadOne', None) is not None:
      leads.append(radar_state.leadOne)
    if getattr(radar_state, 'leadTwo', None) is not None:
      leads.append(radar_state.leadTwo)

    # get lane z array fallback
    lane_z = None
    try:
      if len(self._lane_lines) > 2 and self._lane_lines[2].raw_points.size:
        lane_z = self._lane_lines[2].raw_points[:, 2]
    except Exception:
      lane_z = None

    for l in leads:
      try:
        if not getattr(l, 'status', False):
          continue

        dRel = float(getattr(l, 'dRel', 0.0))
        yRel = float(getattr(l, 'yRel', 0.0))
        v = float(getattr(l, 'vRel', 0.0))  # in m/s (approx)
        v_lat = float(getattr(l, 'vLat', 0.0)) if hasattr(l, 'vLat') else 0.0

        # only render if in front and beyond small threshold
        if dRel <= 2.5:
          # optionally render star for very close when verbosity high
          if s.show_radar_info >= 3:
            # try to project the point to screen for location
            pt = self._map_to_screen(dRel, -yRel, 0.0 + self._path_offset_z)
            if pt:
              rl.draw_text("*", int(pt[0]), int(pt[1]), 40, rl.BLACK)
          continue

        # z from lane if available
        z = 0.0
        if lane_z is not None:
          # find index for dRel
          idx = self._get_path_length_idx(self._lane_lines[2].raw_points[:, 0], dRel)
          if idx < len(lane_z):
            z = lane_z[idx] - 0.61

        side = self._map_to_screen(dRel, -yRel, z + self._path_offset_z)
        if not side:
          continue
        x, y = side

        # speed magnitude
        v_abs = np.sqrt(v * v + v_lat * v_lat)
        v_sum = v_abs if v >= 0.0 else -v_abs

        # predicted future
        t = radar_lat_factor
        if v_abs > 3.0:
          a_dRel = dRel + v * t
          if a_dRel < 2.0:
            a_dRel = 2.0
          a_yRel = yRel + v_lat * t
          a_side = self._map_to_screen(a_dRel, -a_yRel, z + self._path_offset_z)
          if a_side:
            ax, ay = a_side
            # draw line from current to predicted
            rl.draw_line(int(x), int(y), int(ax), int(ay), rl.Color(0, 255, 0, 255) if v_sum > 0 else rl.Color(255, 0, 0, 255))
            # draw predicted circle
            rl.draw_circle(int(ax), int(ay), 10, rl.Color(0, 255, 0, 255) if v_sum > 0 else rl.Color(255, 0, 0, 255))

        # draw speed box
        # convert to display units
        MS_TO_KPH = 3.6
        MS_TO_MPH = 2.2369362920544
        if s.is_metric:
          disp_speed = v_sum * MS_TO_KPH
        else:
          disp_speed = v_sum * MS_TO_MPH
        speed_str = f"{int(round(disp_speed))}"
        # compute box width
        wStr = 35 * max(1, len(speed_str))
        box_x = int(x - wStr / 2)
        box_y = int(y - 35)
        # choose box color
        model_prob = float(getattr(l, 'modelProb', 0.0))
        radar_flag = bool(getattr(l, 'radar', False)) if hasattr(l, 'radar') else True
        if not radar_flag:
          box_color = rl.Color(0, 122, 255, 200)  # blue-ish
        elif abs(model_prob - 0.01) < 1e-6:
          box_color = rl.Color(0, 255, 0, 200)
        else:
          box_color = rl.Color(255, 165, 0, 200) if v_sum > 0 else rl.Color(255, 0, 0, 200)

        rl.draw_rectangle(box_x, box_y, wStr, 42, box_color)
        # draw text (fallback to white)
        try:
          rl.draw_text(speed_str, int(x - (len(speed_str) * 6)), int(y - 10), 40, rl.Color(255, 255, 255, 255))
        except Exception:
          # some pyray wrappers expect different args
          pass

        # additional info when verbosity >=2
        if s.show_radar_info >= 2:
          dist_text = f"{dRel:.1f}" if s.is_metric else f"{dRel * 0.621371:.1f}"
          lat_text = f"{yRel:.1f}"
          try:
            rl.draw_text(lat_text, int(x - 12), int(y - 40), 30, rl.Color(255, 255, 255, 255))
            rl.draw_text(dist_text, int(x - 12), int(y + 30), 30, rl.Color(255, 255, 255, 255))
          except Exception:
            pass

      except Exception:
        # keep drawing others even if one fails
        continue

  @staticmethod
  def _get_path_length_idx(pos_x_array: np.ndarray, path_distance: float) -> int:
    """Get the index corresponding to the given path distance"""
    if len(pos_x_array) == 0:
      return 0
    indices = np.where(pos_x_array <= path_distance)[0]
    return indices[-1] if indices.size > 0 else 0

  def _map_to_screen(self, in_x, in_y, in_z):
    """Project a point in car space to screen space"""
    input_pt = np.array([in_x, in_y, in_z])
    pt = self._car_space_transform @ input_pt

    if abs(pt[2]) < 1e-6:
      return None

    x, y = pt[0] / pt[2], pt[1] / pt[2]

    clip = self._clip_region
    if not (clip.x <= x <= clip.x + clip.width and clip.y <= y <= clip.y + clip.height):
      return None

    return (x, y)

  def _map_line_to_polygon(self, line: np.ndarray, y_off: float, z_off: float, max_idx: int, max_distance: float, allow_invert: bool = True) -> np.ndarray:
    """Convert 3D line to 2D polygon for rendering."""
    if line.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    # Slice points and filter non-negative x-coordinates
    points = line[:max_idx + 1]

    # Interpolate around max_idx so path end is smooth (max_distance is always >= p0.x)
    if 0 < max_idx < line.shape[0] - 1:
      p0 = line[max_idx]
      p1 = line[max_idx + 1]
      x0, x1 = p0[0], p1[0]
      interp_y = np.interp(max_distance, [x0, x1], [p0[1], p1[1]])
      interp_z = np.interp(max_distance, [x0, x1], [p0[2], p1[2]])
      interp_point = np.array([max_distance, interp_y, interp_z], dtype=points.dtype)
      points = np.concatenate((points, interp_point[None, :]), axis=0)

    points = points[points[:, 0] >= 0]
    if points.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    N = points.shape[0]
    # Generate left and right 3D points in one array using broadcasting
    offsets = np.array([[0, -y_off, z_off], [0, y_off, z_off]], dtype=np.float32)
    points_3d = points[None, :, :] + offsets[:, None, :]  # Shape: 2xNx3
    points_3d = points_3d.reshape(2 * N, 3)  # Shape: (2*N)x3

    # Transform all points to projected space in one operation
    proj = self._car_space_transform @ points_3d.T  # Shape: 3x(2*N)
    proj = proj.reshape(3, 2, N)
    left_proj = proj[:, 0, :]
    right_proj = proj[:, 1, :]

    # Filter points where z is sufficiently large
    valid_proj = (np.abs(left_proj[2]) >= 1e-6) & (np.abs(right_proj[2]) >= 1e-6)
    if not np.any(valid_proj):
      return np.empty((0, 2), dtype=np.float32)

    # Compute screen coordinates
    left_screen = left_proj[:2, valid_proj] / left_proj[2, valid_proj][None, :]
    right_screen = right_proj[:2, valid_proj] / right_proj[2, valid_proj][None, :]

    # Define clip region bounds
    clip = self._clip_region
    x_min, x_max = clip.x, clip.x + clip.width
    y_min, y_max = clip.y, clip.y + clip.height

    # Filter points within clip region
    left_in_clip = (
      (left_screen[0] >= x_min) & (left_screen[0] <= x_max) &
      (left_screen[1] >= y_min) & (left_screen[1] <= y_max)
    )
    right_in_clip = (
      (right_screen[0] >= x_min) & (right_screen[0] <= x_max) &
      (right_screen[1] >= y_min) & (right_screen[1] <= y_max)
    )
    both_in_clip = left_in_clip & right_in_clip

    if not np.any(both_in_clip):
      return np.empty((0, 2), dtype=np.float32)

    # Select valid and clipped points
    left_screen = left_screen[:, both_in_clip]
    right_screen = right_screen[:, both_in_clip]

    # Handle Y-coordinate inversion on hills
    if not allow_invert and left_screen.shape[1] > 1:
      y = left_screen[1, :]  # y-coordinates
      keep = y == np.minimum.accumulate(y)
      if not np.any(keep):
        return np.empty((0, 2), dtype=np.float32)
      left_screen = left_screen[:, keep]
      right_screen = right_screen[:, keep]

    return np.vstack((left_screen.T, right_screen[:, ::-1].T)).astype(np.float32)

  @staticmethod
  def _hsla_to_color(h, s, l, a):
    rgb = colorsys.hls_to_rgb(h, l, s)
    return rl.Color(
      int(rgb[0] * 255),
      int(rgb[1] * 255),
      int(rgb[2] * 255),
      int(a * 255)
    )

  @staticmethod
  def _blend_colors(begin_colors, end_colors, t):
    if t >= 1.0:
      return end_colors
    if t <= 0.0:
      return begin_colors

    inv_t = 1.0 - t
    return [rl.Color(
      int(inv_t * start.r + t * end.r),
      int(inv_t * start.g + t * end.g),
      int(inv_t * start.b + t * end.b),
      int(inv_t * start.a + t * end.a)
    ) for start, end in zip(begin_colors, end_colors, strict=True)]
