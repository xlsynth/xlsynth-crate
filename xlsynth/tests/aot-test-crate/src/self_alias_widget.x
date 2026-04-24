// SPDX-License-Identifier: Apache-2.0

struct WidgetHandle {
    widget_id: u8,
}

pub fn echo_widget(widget: WidgetHandle) -> WidgetHandle {
    widget
}
