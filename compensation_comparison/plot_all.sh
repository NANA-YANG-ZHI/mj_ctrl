python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/frictionless" \
    --plot-dir "$SCRIPT_DIR/plots/frictionless"

python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/friction_0.7" \
    --plot-dir "$SCRIPT_DIR/plots/friction_0.7"

echo ""
echo "=========================================================="
echo " Done. Plots saved to:"
echo "   $SCRIPT_DIR/plots/frictionless/compensation_comparison.png"
echo "   $SCRIPT_DIR/plots/friction_0.7/compensation_comparison.png"
echo "=========================================================="