document.getElementById('diffBtn').onclick = function() {
  if(window.rubyEngine)
    document.getElementById('diffArea').innerText =
      window.rubyEngine.two_state.diff_states();
};
document.getElementById('validateBtn').onclick = function() {
  if(window.rubyEngine)
    alert(window.rubyEngine.two_state.validate_blue() ? "Valid!" : "Failed!");
};
document.getElementById('promoteBtn').onclick = function() {
  if(window.rubyEngine)
    window.rubyEngine.two_state.promote_blue_to_green();
};
document.getElementById('rollbackBtn').onclick = function() {
  if(window.rubyEngine)
    window.rubyEngine.two_state.rollback_blue();
};

