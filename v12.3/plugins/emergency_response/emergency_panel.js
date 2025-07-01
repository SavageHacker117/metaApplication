document.getElementById('triggerBtn').onclick = function() {
  let type = document.getElementById('etype').value;
  let loc = JSON.parse(document.getElementById('eloc').value);
  if(window.rubyEngine) window.rubyEngine.plugins.emergency_response.trigger_event(type, loc, 2.0);
};
document.getElementById('branchBtn').onclick = function() {
  if(window.rubyEngine) window.rubyEngine.plugins.emergency_response.branch_disaster(0);
};
document.getElementById('rollbackBtn').onclick = function() {
  if(window.rubyEngine) window.rubyEngine.plugins.emergency_response.rollback_state(0);
};
